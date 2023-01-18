import logging
import math
import warnings
from os import PathLike
from pathlib import Path
from typing import Any, Optional, Callable

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.utils.data
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel, StableDiffusionPipeline
from omegaconf import DictConfig, OmegaConf, ListConfig
from torch import nn
from torch_ema import ExponentialMovingAverage
from transformers import CLIPTokenizer, CLIPTextModel

from modules.clip import hook_forward
from modules.config import OPTIM_TARGETS_DIR
from modules.custom_embeddings import CustomEmbeddingsHook
from modules.dataset import get_dataset, get_sampler, collate_fn
from modules.lora import get_lora
from modules.utils import get_class, physical_core_count, load_state_dict

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


def load_df_pipeline(path: str, vae: Optional[str] = None, tokenizer: Optional[str] = None):
    unet = UNet2DConditionModel.from_pretrained(path, subfolder="unet")

    if vae is None:
        vae = AutoencoderKL.from_pretrained(path, subfolder="vae")
    else:
        vae = AutoencoderKL.from_pretrained(vae)

    text_encoder = CLIPTextModel.from_pretrained(path, subfolder="text_encoder")

    if tokenizer is None:
        tokenizer = CLIPTokenizer.from_pretrained(path, subfolder="tokenizer")
    else:
        tokenizer = CLIPTokenizer.from_pretrained(tokenizer)

    noise_scheduler = DDIMScheduler.from_pretrained(path, subfolder="scheduler")

    return unet, vae, text_encoder, tokenizer, noise_scheduler


def load_ldm_checkpoint(path: Path, config: DictConfig, vae_path: Optional[str | PathLike] = None,
                        tokenizer: Optional[str] = None):
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

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14" if tokenizer is None else tokenizer)

    noise_scheduler = DDIMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

    return unet, vae, text_encoder, tokenizer, noise_scheduler


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


def set_submodule(module: nn.Module, name: str, sub: nn.Module):
    segments = name.split(".")
    module = module.get_submodule(".".join(segments[:-1]))
    module.__setattr__(segments[-1], sub)


def apply_module_config(module: nn.Module, module_configs: ListConfig,
                        fn: Callable[[nn.Module, DictConfig, str], None], recursive=True, path=""):
    for module_config in module_configs:
        index = module_config.get("index")
        targets = module_config.get("targets")

        def invoke_on_submodule(_submodule: nn.Module, _module_path: str):
            _path = _module_path if path == "" else f"{path}.{_module_path}"
            if recursive and targets is not None:
                apply_module_config(_submodule, module_config.targets, fn,
                                    path=_path)
            else:
                fn(_submodule, module_config, _path)

        if index is None:
            for name, submodule in module.named_children():
                if submodule == module:
                    continue

                invoke_on_submodule(submodule, name)
        else:
            for module_path in index:
                submodule = module.get_submodule(module_path)
                invoke_on_submodule(submodule, module_path)


def config_module(config: DictConfig, module: nn.Module):
    if config.get("all", False):
        module.requires_grad_(True)
        return list(module.parameters())

    params_to_optimize = list[nn.Parameter]()

    def apply_innermost(submodule: nn.Module, submodule_config: DictConfig, module_path: str):
        if (lora_config := submodule_config.get("lora")) is not None:
            assert isinstance(submodule, nn.Linear) or isinstance(submodule, nn.Conv2d)
            submodule = get_lora(submodule, **lora_config)
            set_submodule(module, module_path, submodule)
            params_to_optimize.extend([submodule.lora_A, submodule.lora_B])
        else:
            params_to_optimize.extend(submodule.parameters())

    apply_module_config(module, config.targets, apply_innermost)

    for param in params_to_optimize:
        param.requires_grad = True

    if len(list(module.parameters())) != len(params_to_optimize):
        warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True. Gradients will be None")

    return params_to_optimize


def raise_if_nan(x: torch.Tensor, name: str):
    if not torch.any(torch.isnan(x)):
        return

    raise Exception(f"NaN element discovered in {name}")


class StableDiffusionModel(pl.LightningModule):
    unet_ema: Optional[ExponentialMovingAverage] = None

    def __init__(self,
                 config: DictConfig,
                 unet: UNet2DConditionModel,
                 vae: AutoencoderKL,
                 text_encoder: CLIPTextModel,
                 tokenizer: CLIPTokenizer,
                 noise_scheduler: DDIMScheduler):
        super().__init__()

        vae.requires_grad_(False)
        vae.eval()

        text_encoder.requires_grad_(False)
        unet.requires_grad_(False)

        self.params_to_optimize = self._config_net(config.optim_target, unet, text_encoder)

        if config.gradient_checkpointing:
            unet.enable_gradient_checkpointing()
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
            unet, vae, text_encoder, tokenizer, noise_scheduler = \
                load_ldm_checkpoint(model_path, get_ldm_config(config.ldm_config), config.vae, config.tokenizer)
        else:
            unet, vae, text_encoder, tokenizer, noise_scheduler = \
                load_df_pipeline(config.model, config.vae, config.tokenizer)

        logger.info("Weights loaded")

        hook_forward(text_encoder, -config.clip_stop_at_layer)

        if config.custom_embeddings.enabled:
            custom_embeddings_hooker = CustomEmbeddingsHook(config.custom_embeddings.path)
            custom_embeddings_hooker.hook_clip(text_encoder, tokenizer)
            embs = custom_embeddings_hooker.embs
            info = [f"{k}: {v.shape[0]}" for k, v in embs.items()]
            logger.info(f"Loaded {len(embs)} custom embeddings: {info}")

        config.optim_target = OmegaConf.load(OPTIM_TARGETS_DIR / (config.optim_target + ".yaml"))

        return cls(config, unet, vae, text_encoder, tokenizer, noise_scheduler)

    @staticmethod
    def _config_net(config: DictConfig, unet: UNet2DConditionModel, text_encoder: CLIPTextModel):
        params_to_optimize = list[nn.Parameter]()

        if (unet_config := config.get("unet")) is not None:
            params_to_optimize.extend(config_module(unet_config, unet))
        else:
            unet.eval()

        if (te_config := config.get("text_encoder")) is not None:
            params_to_optimize.extend(config_module(te_config, text_encoder))
        else:
            text_encoder.eval()

        return params_to_optimize

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
        if "latents" in batch:
            latents = batch["latents"].to(self.unet.dtype)
        else:
            latents = self._vae_encode(batch["images"]).to(self.unet.dtype)

        raise_if_nan(latents, "VAE output")

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
        if "conds" in batch:
            conds = batch["conds"].to(self.unet.dtype)
        else:
            conds = self._get_embedding(batch["token_ids"]).to(self.unet.dtype)

        raise_if_nan(conds, "text encoder output")

        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, conds).sample

        raise_if_nan(noise_pred, "UNet output")

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

    def train_dataloader(self, use_cache=True):
        use_cache = self.config.data.cache is not None and use_cache
        train_dataset = get_dataset(self.config, self.tokenizer, use_cache)

        sampler = get_sampler(train_dataset, self.config, self.trainer.world_size, self.trainer.global_rank)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            sampler=sampler,
            batch_size=self.config.batch_size,
            collate_fn=collate_fn,
            num_workers=physical_core_count() if self.config.num_workers is None else self.config.num_workers,
            persistent_workers=self.config.num_workers is None or self.config.num_workers > 0
        )
        return train_dataloader

    def configure_optimizers(self):
        optimizer = get_optimizer(self.params_to_optimize, self.config, self.trainer)
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
        if hasattr(self.config.optim_target, "text_encoder"):
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

        if hasattr(self.config.optim_target, "text_encoder") is not None:
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
