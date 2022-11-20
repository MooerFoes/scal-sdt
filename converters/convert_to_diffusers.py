from urllib.request import urlopen

import click
import torch
from diffusers import UNet2DConditionModel, AutoencoderKL, StableDiffusionPipeline
from omegaconf import OmegaConf
from transformers import CLIPTokenizer, CLIPTextModel

from converters.modules.common import get_module_state_dict, DTYPE_CHOICES
from converters.modules.sd_to_diffusers import create_unet_diffusers_config, create_vae_diffusers_config, \
    convert_ldm_clip_checkpoint, create_diffusers_scheduler

V1_INFERENCE = 'https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml'


def get_default_config():
    with urlopen(V1_INFERENCE) as f:
        content = f.read().decode()

    return OmegaConf.create(content)


@click.command()
@click.argument("checkpoint", type=click.File("r"), help="Path to the checkpoint.")
@click.argument("sd-output", type=click.Path(), help="Path to the output Diffusers saved pipeline.")
@click.option("--config", type=click.File("r"), help="Path to the LDM config. (Default: v1-inference.yaml)")
@click.option("--unet-dtype",
              type=DTYPE_CHOICES,
              default="fp32",
              help="Save unet weights in this data type.")
@click.option("--vae-dtype",
              type=DTYPE_CHOICES,
              default="fp32",
              help="Save VAE weights in this data type. (other than fp32 NOT RECOMMENDED)")
@click.option("--text-encoder-dtype",
              type=DTYPE_CHOICES,
              default="fp32",
              help="Save text encoder weights in data type.")
def main(checkpoint, sd_output, config, unet_dtype, vae_dtype, text_encoder_dtype):
    if config is None:
        config = get_default_config()
    else:
        config = OmegaConf.load(config)

    state_dict = torch.load(checkpoint)["state_dict"]

    unet_dict = get_module_state_dict(state_dict, "unet", unet_dtype)
    vae_dict = get_module_state_dict(state_dict, "vae", vae_dtype)
    text_encoder_dict = get_module_state_dict(state_dict, "text_encoder", text_encoder_dtype)

    unet_config = create_unet_diffusers_config(config)
    vae_config = create_vae_diffusers_config(config)

    unet = UNet2DConditionModel(**unet_config)
    unet.load_state_dict(unet_dict)

    vae = AutoencoderKL(**vae_config)
    vae.load_state_dict(vae_dict)

    text_encoder = CLIPTextModel()
    text_encoder.load_state_dict(text_encoder_dict)

    scheduler = create_diffusers_scheduler(config)

    text_model = convert_ldm_clip_checkpoint(checkpoint)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    pipeline = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_model,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler
    )

    pipeline.save_pretrained(sd_output)
