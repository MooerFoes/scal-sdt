from os import PathLike
from pathlib import Path
from typing import Any, Optional

import pytorch_lightning as pl
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline
from pytorch_lightning.utilities import rank_zero_only
from tqdm import tqdm

from .controlnet.control_diffusion import ControlDiffusionPipeline
from .model import LatentDiffusionModel


class SampleCallback(pl.Callback):
    def __init__(self, sample_save_dir: str | PathLike):
        self.sample_dir = Path(sample_save_dir)
        self.pipeline: Optional[StableDiffusionPipeline] = None

    def setup(self, trainer: pl.Trainer, pl_module: LatentDiffusionModel, stage: str):
        args = pl_module.vae, pl_module.condition_model.encoder, pl_module.condition_model.tokenizer, \
            pl_module.unet, pl_module.scheduler, None, None
        kwargs = {"requires_safety_checker": False}

        if pl_module.controlnet is None:
            pipeline = StableDiffusionPipeline(*args, **kwargs)
        else:
            pipeline = ControlDiffusionPipeline(*args, controlnets=[pl_module.controlnet], **kwargs)

        pipeline.set_progress_bar_config(disable=True)
        self.pipeline = pipeline

    @torch.inference_mode()
    @rank_zero_only
    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: LatentDiffusionModel, outputs,
                           batch: Any, batch_idx: int) -> None:
        sampling_config = pl_module.config.get("sampling")
        global_step = trainer.global_step

        if (sampling_config is None or
            not any(sampling_config.concepts) or
            global_step % sampling_config.interval_steps != 0):
            return

        batch_size = sampling_config.batch_size

        save_dir = self.sample_dir / str(global_step)
        save_dir.mkdir(parents=True, exist_ok=True)

        samples = dict[str, list[Image.Image]]()

        text_encoder_training = pl_module.condition_model.training
        pl_module.condition_model.eval()
        pl_module.unet.eval()

        for concept in tqdm(sampling_config.concepts, unit="concept"):
            generator = torch.Generator(device=self.pipeline.device).manual_seed(concept.seed)

            concept_samples = list[Image.Image]()
            i = concept.num_samples
            with tqdm(total=concept.num_samples + (concept.num_samples % batch_size),
                      desc="Generating samples") as progress:
                while True:
                    actual_bsz = i if i - batch_size < 0 else batch_size

                    if actual_bsz <= 0:
                        break

                    concept_samples.extend(
                        self.pipeline(
                            num_images_per_prompt=actual_bsz,
                            generator=generator,
                            prompt=concept.prompt,
                            negative_prompt=concept.negative_prompt,
                            num_inference_steps=concept.steps,
                            guidance_scale=concept.cfg_scale,
                            width=concept.width,
                            height=concept.height,
                        ).images
                    )
                    progress.update(actual_bsz)

                    i -= actual_bsz
            samples[concept.prompt] = concept_samples

        pl_module.condition_model.train(text_encoder_training)
        pl_module.unet.train()

        for i, (_, images) in enumerate(samples.items()):
            for j, image in enumerate(images):
                image.save(save_dir / f"{i}-{j}.png")

        wandb_config = pl_module.config.loggers.get("wandb")

        if (wandb_config is not None and
            wandb_config.get("sample", False) and
            any(samples)):
            import wandb
            log_samples = {
                "samples": {
                    prompt[:230]: [wandb.Image(x) for x in images] for prompt, images in samples.items()
                }
            }
            wandb.log(log_samples, global_step)
