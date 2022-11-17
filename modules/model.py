import itertools
import math
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.utils.data
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTokenizer

from modules.clip import CLIPWithSkip


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


def load_model(config):
    unet = UNet2DConditionModel.from_pretrained(config.model, subfolder="unet")
    vae = AutoencoderKL.from_pretrained(config.vae if config.vae else config.model, subfolder="vae")
    text_encoder = CLIPWithSkip.from_pretrained(config.model, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(config.tokenizer if config.tokenizer else config.model,
                                              subfolder="tokenizer")
    noise_scheduler = DDIMScheduler.from_config(config.model, subfolder="scheduler")

    text_encoder.stop_at_layer = config.clip_stop_at_layer

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

        if config.gradient_checkpointing:
            unet.enable_gradient_checkpointing()
            if config.train_text_encoder:
                text_encoder.gradient_checkpointing_enable()

        self.unet.set_use_memory_efficient_attention_xformers(True)

    @torch.inference_mode()
    def vae_encode(self, image):
        return self.vae.encode(image).latent_dist.sample() * 0.18215

    # @rank_zero_only
    # def log_samples(self):
    #     pipeline = pipeline.to(accelerator.device)
    #     pipeline.set_progress_bar_config(disable=True)
    #     pipeline.enable_xformers_memory_efficient_attention()
    #     sample_dir = run_output_dir / "samples"
    #     sample_dir.mkdir(exist_ok=True)
    #     samples = []
    #     with torch.autocast("cuda"), torch.inference_mode():
    #         for concept in tqdm(config.sampling.concepts, unit="concept"):
    #             g_cuda = torch.Generator(device=accelerator.device).manual_seed(concept.seed)
    #             concept_samples = []
    #             with tqdm(total=concept.num_samples + (concept.num_samples % config.sampling.batch_size),
    #                       desc=f"Generating samples") as progress:
    #
    #                 for _ in range(math.ceil(concept.num_samples / config.sampling.batch_size)):
    #                     concept_samples.extend(pipeline(
    #                         prompt=concept.prompt,
    #                         negative_prompt=concept.negative_prompt,
    #                         guidance_scale=concept.cfg_scale,
    #                         num_inference_steps=concept.steps,
    #                         num_images_per_prompt=config.sampling.batch_size,
    #                         generator=g_cuda).images)
    #                     progress.update(config.sampling.batch_size)
    #             samples.append((concept.prompt, concept_samples))
    #     del pipeline
    #     if torch.cuda.is_available():
    #         torch.cuda.empty_cache()
    #
    #     for i, (_, images) in enumerate(samples):
    #         for j, image in enumerate(images):
    #             image.save(sample_dir / f"{global_steps}_{i}_{j}.png")
    #
    #     if wandb_enabled and config.monitoring.wandb.sample and any(samples):
    #         log_samples = {"samples": {prompt: [wandb.Image(x) for x in images] for prompt, images in samples}}
    #         accelerator.log(log_samples, global_steps, {"commit": False})

    def training_step(self, batch, batch_idx):
        # if batch.latents is not None:
        #     latents = batch.latents
        # else:
        #     latents = self.vae_encode(batch["images"])
        latents = self.vae_encode(batch["images"])

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        # if self.config.train_text_encoder:
        #     conds = self.text_encoder.forward(batch["token_ids"])
        # else:
        #     conds = batch.conds
        with torch.no_grad():
            conds = self.text_encoder.forward(batch["token_ids"]).last_hidden_state

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
        return [optimizer], [lr_scheduler]
