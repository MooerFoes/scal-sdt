import itertools
import logging
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.utils.data
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel, StableDiffusionPipeline
from omegaconf import DictConfig, OmegaConf
from torch_ema import ExponentialMovingAverage
from transformers import CLIPTokenizer, CLIPTextModel

from modules.clip import hook_forward
from modules.convert.common import load_state_dict
from modules.dataset import get_dataset, collate_fn, get_sampler
from modules.utils import get_class, physical_core_count

logger = logging.getLogger()


def get_optimizer(paramters_to_optimize, config, trainer: pl.Trainer):
    params = dict(config.optimizer.params)

    lr_scale_config = config.optimizer.lr_scale
    if lr_scale_config.enabled:
        coeff = trainer.accumulate_grad_batches * config.batch_size * trainer.num_nodes * trainer.num_devices
        if lr_scale_config.method == "linear":
            params["lr"] *= coeff
        elif lr_scale_config.method == "sqrt":
            params["lr"] *= math.sqrt(coeff)
        else:
            raise ValueError(lr_scale_config.method)

    optimizer_class = get_class(config.optimizer.name)

    if "beta1" in params and "beta2" in params:
        params["betas"] = (params["beta1"], params["beta2"])
        del params["beta1"]
        del params["beta2"]

    optimizer = optimizer_class(paramters_to_optimize, **params)

    return optimizer


def get_lr_scheduler(config, optimizer) -> Any:
    lr_sched_config = config.optimizer.lr_scheduler
    scheduler = get_class(lr_sched_config.name)(optimizer, **lr_sched_config.params)

    if lr_sched_config.warmup.enabled:
        from .warmup_lr import WarmupLR
        scheduler = WarmupLR(scheduler,
                             init_lr=lr_sched_config.warmup.init_lr,
                             num_warmup=lr_sched_config.warmup.steps,
                             warmup_strategy=lr_sched_config.warmup.strategy)

    return scheduler


def load_df_pipeline(path: str, vae: Optional[str] = None):
    unet = UNet2DConditionModel.from_pretrained(path, subfolder="unet")

    if vae is None:
        vae = AutoencoderKL.from_pretrained(path, subfolder="vae")
    else:
        vae = AutoencoderKL.from_pretrained(vae)

    text_encoder = CLIPTextModel.from_pretrained(path, subfolder="text_encoder")

    return unet, vae, text_encoder


def load_ldm_checkpoint(path: Path, config: DictConfig, vae_path: Optional[Path] = None):
    state_dict = load_state_dict(path)

    from modules.convert.sd_to_diffusers import (
        create_unet_diffusers_config,
        convert_ldm_unet_checkpoint,
        create_vae_diffusers_config,
        convert_ldm_vae_checkpoint,
        convert_ldm_clip_checkpoint
    )

    unet_config = create_unet_diffusers_config(config)
    unet_state_dict = convert_ldm_unet_checkpoint(state_dict, unet_config)
    unet = UNet2DConditionModel(**unet_config)
    unet.load_state_dict(unet_state_dict)

    vae_config = create_vae_diffusers_config(config)
    converted_vae_checkpoint = convert_ldm_vae_checkpoint(state_dict, vae_config, vae_path)
    vae = AutoencoderKL(**vae_config)
    vae.load_state_dict(converted_vae_checkpoint)

    text_encoder = convert_ldm_clip_checkpoint(state_dict)

    return unet, vae, text_encoder


def get_ldm_config(link_or_path: str):
    if link_or_path.startswith("http://") or link_or_path.startswith("https://"):
        import requests
        with requests.Session() as session:
            config_str = session.get(link_or_path).content.decode("utf-8")
    elif Path(link_or_path).exists():
        with open(link_or_path, "r") as f:
            config_str = f.read()
    else:
        raise ValueError(f'"{link_or_path}" is not a valid link or path')

    return OmegaConf.create(config_str)


class StableDiffusionModel(pl.LightningModule):
    unet_ema: Optional[ExponentialMovingAverage]

    def __init__(self,
                 config: DictConfig,
                 unet: UNet2DConditionModel,
                 vae: AutoencoderKL,
                 text_encoder: CLIPTextModel,
                 tokenizer: CLIPTokenizer,
                 noise_scheduler: DDIMScheduler):
        super().__init__()

        vae.requires_grad_(False)
        if not config.train_text_encoder:
            text_encoder.requires_grad_(False)
            self._text_encode_context_cls = torch.no_grad
        else:
            self._text_encode_context_cls = nullcontext

        if config.gradient_checkpointing:
            unet.enable_gradient_checkpointing()
            if config.train_text_encoder:
                text_encoder.gradient_checkpointing_enable()

        if config.xformers:
            unet.set_use_memory_efficient_attention_xformers(True)

        self.pipeline = StableDiffusionPipeline(vae, text_encoder, tokenizer, unet, noise_scheduler, None, None, False)
        self.pipeline.set_progress_bar_config(disable=True)

        self.config = config
        self.unet = unet
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.noise_scheduler = noise_scheduler

        self.save_hyperparameters(config)

    @classmethod
    def from_config(cls, config: DictConfig):
        if (model_path := Path(config.model)).suffix.lower() == ".ckpt":
            unet, vae, text_encoder = \
                load_ldm_checkpoint(model_path, get_ldm_config(config.ldm_config), Path(config.vae))
        else:
            unet, vae, text_encoder = \
                load_df_pipeline(config.model, config.vae)

        logger.info("Weights loaded")

        if config.tokenizer is None:
            tokenizer = CLIPTokenizer.from_pretrained(config.model, subfolder="tokenizer")
        else:
            tokenizer = CLIPTokenizer.from_pretrained(config.tokenizer)

        hook_forward(text_encoder, -config.clip_stop_at_layer)
        noise_scheduler = DDIMScheduler.from_pretrained(config.model, subfolder="scheduler")

        return cls(config, unet, vae, text_encoder, tokenizer, noise_scheduler)

    @torch.no_grad()
    def _vae_encode(self, image):
        device = self.unet.device

        if self.config.med_vram:
            self.unet.to("cpu")

        latents = self.vae.encode(image).latent_dist.sample() * 0.18215

        if self.config.med_vram:
            self.unet.to(device)

        return latents

    def _encode_token_ids(self, token_ids: torch.Tensor):
        with self._text_encode_context_cls():
            return self.text_encoder.forward(token_ids).last_hidden_state

    def _get_embedding(self, token_ids: torch.Tensor):
        uc_conf = self.config.uncond

        if not (uc_conf.enabled and torch.rand(1) < uc_conf.p):
            return self._encode_token_ids(token_ids)

        bsz, length = token_ids.shape
        encoder_config = self.text_encoder.config

        match uc_conf.cond:
            case "zeros":
                return torch.zeros(bsz, length, encoder_config.hidden_size, device=self.unet.device)
            case "bos":
                fill_token_id = encoder_config.bos_token_id
            case "eos":
                fill_token_id = encoder_config.eos_token_id
            case _:
                raise Exception("Invalid uncond.cond")

        token_ids = torch.full((bsz, length), fill_token_id, device=self.unet.device)

        return self.text_encoder.forward(token_ids).last_hidden_state

    def training_step(self, batch, batch_idx):
        # if batch.latents is not None:
        #     latents = batch.latents
        # else:
        #     latents = self.vae_encode(batch["images"])
        latents = self._vae_encode(batch["images"]).to(self.unet.dtype)

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,),
                                  dtype=torch.int64, device=latents.device)

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        # if self.config.train_text_encoder:
        #     conds = self.text_encoder.forward(batch["token_ids"])
        # else:
        #     conds = batch.conds
        conds = self._get_embedding(batch["token_ids"]).to(self.unet.dtype)

        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, conds).sample

        if self.config.prior_preservation.enabled:
            # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
            noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
            noise, noise_prior = torch.chunk(noise, 2, dim=0)

            # Compute instance loss
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="none").mean([1, 2, 3]).mean()

            # Compute prior loss
            prior_loss = F.mse_loss(noise_pred_prior.float(), noise_prior.float(), reduction="mean")

            # Add the prior loss to the instance loss.
            loss = loss + self.config.prior_preservation.prior_loss_weight * prior_loss
        else:
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

        self.log_dict({
            "train_loss": loss.item(),
            "lr": self.lr_schedulers().get_lr()[0]
        })
        return loss

    def train_dataloader(self):
        train_dataset = get_dataset(self.config, self.tokenizer)

        sampler = get_sampler(train_dataset, self.config, self.trainer.world_size, self.trainer.global_rank)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            sampler=sampler,
            batch_size=self.config.batch_size,
            collate_fn=collate_fn,
            num_workers=physical_core_count() if self.config.num_workers is None else self.config.num_workers,
            persistent_workers=True
        )
        return train_dataloader

    def configure_optimizers(self):
        params_to_optimize = itertools.chain(self.unet.parameters(), self.text_encoder.parameters()) \
            if self.config.train_text_encoder else self.unet.parameters()
        optimizer = get_optimizer(params_to_optimize, self.config, self.trainer)
        lr_scheduler = get_lr_scheduler(self.config, optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1
            }
        }

    def on_save_checkpoint(self, checkpoint: dict[str, Any]):
        """Diffusers -> SSDT LDM"""
        state_dict = checkpoint["state_dict"]

        # EMA (Cannot be directly loaded by LDM)
        ema_dict = {}
        if self.unet_ema is not None:
            with self.unet_ema.average_parameters(), torch.no_grad():
                ema_dict = {
                    "unet_ema": {
                        "decay": self.unet_ema.decay,
                        "num_updates": self.unet_ema.num_updates,
                        "state_dict": self.unet.state_dict()
                    }
                }
            self.unet_ema.collected_params = None

        from modules.convert.diffusers_to_sd import convert_unet_state_dict

        unet_dict = {k.removeprefix("unet."): v for k, v in state_dict.items() if k.startswith("unet.")}
        unet_dict = convert_unet_state_dict(unet_dict)
        unet_dict = {"model.diffusion_model." + k: v for k, v in unet_dict.items()}

        text_encoder_dict = {}
        # Save text encoder state only if it was unfreezed
        if self.config.train_text_encoder:
            text_encoder_dict = {"cond_stage_model.transformer." + k.removeprefix("text_encoder."): v
                                 for k, v in state_dict.items() if k.startswith("text_encoder.")}

        checkpoint["state_dict"] = {
            **unet_dict,
            **ema_dict,
            **text_encoder_dict
        }

    def on_load_checkpoint(self, checkpoint: dict[str, Any]):
        """SSDT LDM -> Diffusers"""
        state_dict = checkpoint["state_dict"]

        ema_dict = state_dict["unet_ema"]

        from modules.convert.sd_to_diffusers import convert_ldm_unet_checkpoint

        unet_dict = convert_ldm_unet_checkpoint(state_dict, self.unet.config, extract_ema=False)

        text_encoder_dict = {k.removeprefix("cond_stage_model.transformer."): v
                             for k, v in state_dict.items() if k.startswith("cond_stage_model.transformer.")}

        checkpoint["state_dict"] = {
            "unet": unet_dict,
            "unet_ema": ema_dict,
            "text_encoder": text_encoder_dict
        }

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True):
        if self.should_update_ema:
            ema_dict = state_dict["unet_ema"]
            self.unet.load_state_dict(ema_dict["state_dict"])

            self.unet_ema = ExponentialMovingAverage(self.unet.parameters(), ema_dict["decay"])
            self.unet_ema.num_updates = ema_dict["num_updates"]

        self.unet.load_state_dict(state_dict["unet"])

        if self.config.train_text_encoder:
            self.text_encoder.load_state_dict(state_dict["text_encoder"])

    @property
    def should_update_ema(self):
        return self.config.ema.enabled and self.trainer.is_global_zero

    def on_fit_start(self):
        if self.should_update_ema:
            # Stored on RAM
            self.unet_ema = ExponentialMovingAverage(self.unet.parameters(), self.config.ema.decay)

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int):
        # After optimizer step
        if self.should_update_ema:
            self.unet_ema.to(self.unet.device)
            self.unet_ema.update()
            self.unet_ema.to("cpu")

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step(epoch=self.trainer.global_step / self.trainer.num_training_batches)
