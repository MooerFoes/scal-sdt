import itertools
import logging
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.utils.data
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel, StableDiffusionPipeline
from transformers import CLIPTokenizer

from modules.clip import CLIPWithSkip

logger = logging.getLogger()


def get_class(name: str):
    import importlib
    module_name, class_name = name.rsplit(".", 1)
    module = importlib.import_module(module_name, package=None)
    return getattr(module, class_name)


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


def load_df_pipeline(path, vae=None, tokenizer=None, clip_stop_at_layer=1):
    unet = UNet2DConditionModel.from_pretrained(path, subfolder="unet")

    if vae is None:
        vae = AutoencoderKL.from_pretrained(path, subfolder="vae")
    else:
        vae = AutoencoderKL.from_pretrained(vae)

    text_encoder = CLIPWithSkip.from_pretrained(path, subfolder="text_encoder")
    text_encoder.stop_at_layer = clip_stop_at_layer

    if tokenizer is None:
        tokenizer = CLIPTokenizer.from_pretrained(path, subfolder="tokenizer")
    else:
        tokenizer = CLIPTokenizer.from_pretrained(tokenizer)

    return unet, vae, text_encoder, tokenizer


def load_model(config):
    model_path = Path(config.model)

    if (model_path / "model_index.json").is_file():
        unet, vae, text_encoder, tokenizer = \
            load_df_pipeline(model_path, config.vae, config.tokenizer, config.clip_stop_at_layer)
    elif model_path.suffix.lower() == ".ckpt":
        raise NotImplementedError("Loading directly from SD checkpoint is not implemented.")
    else:
        raise ValueError("Invalid model. (Not Diffusers format nor SD format)")

    logger.info("Model loaded")

    noise_scheduler = DDIMScheduler.from_config(config.model, subfolder="scheduler")

    return StableDiffusionModel(config, unet, vae, text_encoder, tokenizer, noise_scheduler)


class StableDiffusionModel(pl.LightningModule):
    def __init__(self,
                 config,
                 unet: UNet2DConditionModel,
                 vae: AutoencoderKL,
                 text_encoder: CLIPWithSkip,
                 tokenizer: CLIPTokenizer,
                 noise_scheduler: DDIMScheduler):
        super().__init__()
        self.noise_scheduler = noise_scheduler
        self.config = config
        self.unet = unet
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer

        self.vae.requires_grad_(False)
        if not config.train_text_encoder:
            self.text_encoder.requires_grad_(False)
            self._text_encode_context = torch.no_grad()
        else:
            self._text_encode_context = nullcontext()

        if config.gradient_checkpointing:
            unet.enable_gradient_checkpointing()
            if config.train_text_encoder:
                text_encoder.gradient_checkpointing_enable()

        self.unet.set_use_memory_efficient_attention_xformers(config.xformers)

        self.pipeline = StableDiffusionPipeline(vae, text_encoder, tokenizer, unet, noise_scheduler)
        self.pipeline.set_progress_bar_config(disable=True)

        self.save_hyperparameters(config)

    @torch.no_grad()
    def _vae_encode(self, image):
        device = self.unet.device

        if self.config.med_vram:
            self.unet.to("cpu")

        latents = self.vae.encode(image).latent_dist.sample() * 0.18215

        if self.config.med_vram:
            self.unet.to(device)

        return latents

    def _encode_token_ids(self, token_ids):
        with self._text_encode_context:
            return self.text_encoder.forward(token_ids).last_hidden_state

    def training_step(self, batch, batch_idx):
        # if batch.latents is not None:
        #     latents = batch.latents
        # else:
        #     latents = self.vae_encode(batch["images"])
        latents = self._vae_encode(batch["images"])

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
        conds = self._encode_token_ids(batch["token_ids"])

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
            "train_loss": loss,
            "lr": self.lr_schedulers().get_lr()[0]
        })
        return loss

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        pass

    def test_step(self, batch, batch_idx):
        pass

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

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step(epoch=self.trainer.global_step / self.trainer.num_training_batches)
