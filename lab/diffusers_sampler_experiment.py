import logging
import sys
from pathlib import Path

import click
import torch
from PIL import Image
from diffusers import UNet2DConditionModel, AutoencoderKL
from diffusers.pipelines import StableDiffusionPipeline
from omegaconf import OmegaConf
from torchvision import transforms
from transformers import CLIPTokenizer

parent = str(Path(__file__).parent.parent.absolute())
sys.path.append(parent)

from modules.clip import CLIPWithSkip

logger = logging.getLogger("exp")
logging.basicConfig(level="INFO")


def get_scheduler():
    # from diffusers.schedulers.scheduling_pndm import PNDMScheduler
    # scheduler = PNDMScheduler(
    #     num_train_timesteps=1000,
    #     beta_start=0.00085,
    #     beta_end=0.0120,
    #     beta_schedule="scaled_linear",
    #     skip_prk_steps=True
    # )

    # from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    # scheduler = DDIMScheduler(
    #     num_train_timesteps=1000,
    #     beta_start=0.00085,
    #     beta_end=0.0120,
    #     beta_schedule="scaled_linear",
    #     clip_sample=False,
    #     set_alpha_to_one=False,
    # )

    # from diffusers.schedulers.scheduling_lms_discrete import LMSDiscreteScheduler
    # scheduler = LMSDiscreteScheduler(
    #     num_train_timesteps=1000,
    #     beta_start=0.00085,
    #     beta_end=0.0120,
    #     beta_schedule="scaled_linear"
    # )

    # from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
    # scheduler = EulerDiscreteScheduler(
    #     num_train_timesteps=1000,
    #     beta_start=0.00085,
    #     beta_end=0.0120,
    #     beta_schedule="scaled_linear"
    # )

    from diffusers.schedulers.scheduling_euler_ancestral_discrete import EulerAncestralDiscreteScheduler
    scheduler = EulerAncestralDiscreteScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.0120,
        beta_schedule="scaled_linear"
    )

    return scheduler


@click.command()
@click.option("--config", required=True)
def main(config):
    default_config = OmegaConf.load('configs/__reserved_default__.yaml')
    config = OmegaConf.merge(default_config, OmegaConf.load(config))

    unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
        config.model,
        subfolder="unet"
    )
    # unet.half()
    unet.set_use_memory_efficient_attention_xformers(True)

    vae: AutoencoderKL = AutoencoderKL.from_pretrained(
        config.vae if config.vae else config.model,
        subfolder="vae",
    )

    text_encoder: CLIPWithSkip = CLIPWithSkip.from_pretrained(
        config.model,
        subfolder="text_encoder",
    )
    text_encoder.stop_at_layer = 1

    tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(
        config.tokenizer if config.tokenizer else config.model,
        subfolder="tokenizer"
    )

    scheduler = get_scheduler()

    pipeline = StableDiffusionPipeline(
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler
    )

    pipeline.to("cuda")
    pipeline.set_progress_bar_config(disable=True)

    SEED = 114514

    generator = torch.Generator(device="cuda")
    generator.manual_seed(SEED)

    output = Path("labout")
    output.mkdir(exist_ok=True)

    imgs = list[Image.Image]()

    def callback(i, t, latents):
        latents = 1 / 0.18215 * latents
        z = vae.decode(latents).sample[0]
        z = (z / 2 + 0.5).clamp(0, 1)
        img: Image = transforms.ToPILImage()(z)
        img.save(output / f"{SEED}-step={i + 1}-timestep={t:.3f}.png")
        imgs.append(img)

    with torch.inference_mode():
        image: Image = pipeline(
            prompt="best quality, 1girl",
            negative_prompt="monochrome, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, "
                            "fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, "
                            "watermark, username, blurry",
            num_inference_steps=20,
            guidance_scale=7,
            num_images_per_prompt=1,
            width=512,
            height=512,
            generator=generator,
            callback=callback,
            callback_steps=1
        ).images[0]

    image.save(output / f"{SEED}-final.png")

    append_images = imgs[1:]
    # Why animated webp is so poor supported
    imgs[0].save(output / f"{SEED}-step={len(imgs)}.gif", save_all=True, append_images=append_images)


if __name__ == '__main__':
    main()
