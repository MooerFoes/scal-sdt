import logging
import math
import warnings
from pathlib import Path
from typing import Any, Optional, Sequence

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.utils.data
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel, StableDiffusionPipeline
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm
from transformers import CLIPTextModel

from modules import text_encoders
from modules.configs import OPTIM_TARGETS_DIR, get_ldm_config
from modules.dataset import get_dataset, get_sampler, collate_fn
from modules.lora import get_lora
from modules.text_encoders import CLIPTextEncoder, CustomEmbedding
from modules.utils import get_class, physical_core_count, load_state_dict, infer_format_from_path, set_submodule, \
    apply_module_config, raise_if_nan

logger = logging.getLogger()


def get_optimizer(params, config, trainer: pl.Trainer):
    optimizer_params = dict(config.optimizer.params)

    if "beta1" in optimizer_params and "beta2" in optimizer_params:
        optimizer_params["betas"] = (optimizer_params["beta1"], optimizer_params["beta2"])
        del optimizer_params["beta1"]
        del optimizer_params["beta2"]

    optimizer_class = get_class(config.optimizer.name)
    optimizer = optimizer_class(params, **optimizer_params)

    lr_scale_config = config.optimizer.lr_scale
    if lr_scale_config.enabled:
        coeff = trainer.accumulate_grad_batches * config.batch_size * trainer.num_nodes * trainer.num_devices
        match lr_scale_config.method:
            case "linear":
                pass
            case "sqrt":
                coeff = math.sqrt(coeff)
            case _:
                raise ValueError(lr_scale_config.method)

        # optimizer.defaults["lr"] *= coeff

        for param_group in optimizer.param_groups:
            if "lr" in param_group:
                param_group["lr"] *= coeff

            if "weight_decay" in param_group:
                param_group["weight_decay"] /= coeff

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


def load_df_pipeline(path: str, vae: Optional[str] = None, clip_stop_at_layer=1):
    unet = UNet2DConditionModel.from_pretrained(path, subfolder="unet")

    if vae is None:
        vae = AutoencoderKL.from_pretrained(path, subfolder="vae")
    else:
        vae = AutoencoderKL.from_pretrained(vae)

    text_encoder = CLIPTextEncoder(path, clip_stop_at_layer)

    scheduler = DDIMScheduler.from_pretrained(path, subfolder="scheduler")

    return unet, vae, text_encoder, scheduler


def load_ldm_checkpoint(path: Path, config: DictConfig, vae_path: Optional[Path] = None, clip_stop_at_layer=1):
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
    vae_state_dict = convert_ldm_vae_checkpoint(state_dict, vae_config, vae_path)
    vae = AutoencoderKL(**vae_config)
    vae.load_state_dict(vae_state_dict)

    clip_state_dict = convert_ldm_clip_checkpoint(state_dict)
    text_encoder = CLIPTextEncoder(text_encoders.CLIP_L, clip_stop_at_layer)
    text_encoder.encoder.load_state_dict(clip_state_dict, strict=False)

    scheduler = DDIMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

    return unet, vae, text_encoder, scheduler


def load_components(name: str, vae: Optional[str] = None, ldm_config_path: Optional[str] = None):
    if (path := Path(name)).is_file():
        return load_ldm_checkpoint(path, get_ldm_config(ldm_config_path), Path(vae) if vae is not None else None)
    else:
        return load_df_pipeline(name, vae)


def config_module(config: DictConfig, module: nn.Module):
    module.requires_grad_(False)
    param_groups = list[dict[str, Any]]()

    def apply_innermost(submodule: nn.Module, submodule_config: DictConfig, module_path: str):
        if (lora_config := submodule_config.get("lora")) is not None:
            assert isinstance(submodule, nn.Linear) or isinstance(submodule, nn.Conv2d)
            submodule = get_lora(submodule, **lora_config)
            set_submodule(module, module_path, submodule)
            params = [submodule.lora_A, submodule.lora_B]
        else:
            params = list(submodule.parameters())

        for param in params:
            param.requires_grad = True

        param_groups.append({
            "params": params,
            **submodule_config.get("optimizer", {})
        })

    apply_module_config(module, config.targets, apply_innermost)

    if len(list(module.parameters())) != len([param
                                              for group in param_groups
                                              for param in group["params"]]):
        warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True. Gradients will be None")

    return param_groups


class LatentDiffusionModel(pl.LightningModule):
    unet_ema: Optional[ExponentialMovingAverage] = None

    def __init__(self,
                 config: DictConfig,
                 unet: UNet2DConditionModel,
                 scheduler,
                 vae: AutoencoderKL,
                 condition_model: CLIPTextEncoder):
        super().__init__()

        vae.requires_grad_(False)
        vae.eval()

        self.param_groups = self._config_net(config.optim_target, unet, condition_model.encoder)

        if config.gradient_checkpointing:
            unet.enable_gradient_checkpointing()
            condition_model.encoder.gradient_checkpointing_enable()

        if config.xformers:
            unet.set_use_memory_efficient_attention_xformers(True)

        self.pipeline = StableDiffusionPipeline(vae, condition_model.encoder, condition_model.tokenizer, unet,
                                                scheduler, None, None, False)
        self.pipeline.set_progress_bar_config(disable=True)

        self.config = config
        self.unet = unet
        self.scheduler = scheduler
        self.vae = vae
        self.condition_model = condition_model

        self.save_hyperparameters(config)

    @classmethod
    def from_config(cls, config: DictConfig):
        unet, vae, text_encoder, scheduler = \
            load_components(config.model, config.vae, config.ldm_config)

        logger.info("Weights loaded")

        if config.custom_embeddings.enabled:
            embed_paths = [path for path in Path(config.custom_embeddings.path).iterdir()
                           if infer_format_from_path(path) is not None]
            embs = [CustomEmbedding.load(embed_path) for embed_path in tqdm(embed_paths)]
            logger.info(f"Loaded total of {len(embs)} custom embeddings")
            text_encoder.init_custom_embeddings(embs)

        if isinstance(config.optim_target, str):
            config.optim_target = OmegaConf.load(OPTIM_TARGETS_DIR / (config.optim_target + ".yaml"))
        else:
            assert isinstance(config.optim_target, DictConfig)

        return cls(config, unet, scheduler, vae, text_encoder)

    @staticmethod
    def _config_net(config: DictConfig, unet: UNet2DConditionModel, text_encoder: CLIPTextModel):
        param_groups = list[dict[str, Any]]()

        def _add_component(component_config: Optional[DictConfig], component: nn.Module):
            params = []

            if component_config is not None:
                params = config_module(component_config, component)

            if not any(params):
                from types import MethodType
                component.requires_grad_(False)
                component.eval()
                component.train = MethodType(lambda self, mode: self, text_encoder)
                return

            param_groups.extend(params)

        _add_component(config.get("unet"), unet)
        _add_component(config.get("text_encoder"), text_encoder)

        return param_groups

    @torch.no_grad()
    def _vae_encode(self, image):
        device = self.unet.device

        if self.config.med_vram:
            self.unet.to("cpu")

        latents = self.vae.encode(image).latent_dist.sample() * 0.18215

        if self.config.med_vram:
            self.unet.to(device)

        return latents

    @torch.no_grad()
    def _get_condition(self, prompts: Sequence[str]):
        uc_conf = self.config.uncond

        if not (uc_conf.enabled and torch.rand(1) < uc_conf.p):
            return self.condition_model(prompts)

        bsz = len(prompts)
        length = self.condition_model.tokenizer.model_max_length

        encoder_config = self.condition_model.config

        match uc_conf.cond:
            case "zeros":
                return torch.zeros(bsz, length, encoder_config.hidden_size, device=self.unet.device)
            case "eos":
                return self.condition_model(bsz * [""])
            case _:
                raise Exception("Invalid uncond.cond")

    def _denoise_loss(self, latents, conds):
        latents.to(self.unet.dtype)
        conds.to(self.unet.dtype)

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (bsz,),
                                  dtype=torch.int64, device=self.unet.device)

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        pred = self.unet(noisy_latents, timesteps, conds).sample

        match self.scheduler.config.prediction_type:
            case "epsilon":
                target = noise
            case "sample":
                target = latents
            case "v":
                target = self.scheduler.get_velocity(latents, noise, timesteps)
            case _:
                raise Exception("Unknown prediction type")

        return F.mse_loss(pred, target, reduction="none")

    def training_step(self, batch, batch_idx):
        if "latents" in batch:
            latents = batch["latents"]
        else:
            latents = self._vae_encode(batch["images"])

        raise_if_nan(latents, "VAE output")

        # Get the text embedding for conditioning
        if "conds" in batch:
            conds = batch["conds"]
        else:
            conds = self._get_condition(batch["prompts"])

        raise_if_nan(conds, "text encoder output")

        loss = self._denoise_loss(latents, conds)

        raise_if_nan(loss, "loss")

        if self.config.prior_preservation.enabled:
            loss, prior_loss = torch.chunk(loss, 2, dim=0)
            loss = loss.mean() + self.config.prior_preservation.prior_loss_weight * prior_loss.mean()
        else:
            loss = loss.mean()

        self.log_dict({
            "train_loss": loss.item(),
            "lr": self.lr_schedulers().get_lr()[0]
        })
        return loss

    def train_dataloader(self, use_cache=True):
        use_cache = self.config.data.cache is not None and use_cache
        train_dataset = get_dataset(self.config, use_cache)

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
        optimizer = get_optimizer(self.param_groups, self.config, self.trainer)
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
            self.condition_model.encoder.load_state_dict(state_dict["text_encoder"])

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
