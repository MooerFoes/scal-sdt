from pathlib import Path

import click
import torch

from modules.convert.common import DTYPE_CHOICES, STATE_DICT, get_module_state_dict
from modules.convert.diffusers_to_sd import convert_unet_state_dict, convert_vae_state_dict


@click.command()
@click.argument("checkpoint", type=click.File("rb"))
@click.argument("output", type=click.Path())
@click.option("--text-encoder/--no-text-encoder", default=True, help="Whether to include text encoder weights.")
@click.option("--vae/--no-vae", default=True, help="Whether to include VAE weights.")
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
@click.option("--overwrite", is_flag=True)
def main(checkpoint, output, text_encoder, vae, unet_dtype, vae_dtype, text_encoder_dtype, overwrite):
    """Converts SCAL-SDT checkpoint to Stable Diffusion format.

    CHECKPOINT: Path to the SCAL-SDT checkpoint.
    OUTPUT: Output Stable Diffusion ckpt path."""
    if Path(output).exists() and not overwrite:
        raise FileExistsError(f"{output} already exists")

    state_dict: STATE_DICT = torch.load(checkpoint)["state_dict"]

    # Convert the UNet model
    unet_dict = get_module_state_dict(state_dict, "unet", unet_dtype)
    unet_dict = convert_unet_state_dict(unet_dict)
    unet_dict = {"model.diffusion_model." + k: v for k, v in unet_dict.items()}

    vae_dict = {}
    if vae:
        # Convert the VAE model
        vae_dict = get_module_state_dict(state_dict, "vae", vae_dtype)
        vae_dict = convert_vae_state_dict(vae_dict)
        vae_dict = {"first_stage_model." + k: v for k, v in vae_dict.items()}

    text_encoder_dict = {}
    if text_encoder:
        # Convert the text encoder model
        text_encoder_dict = get_module_state_dict(state_dict, "text_encoder", text_encoder_dtype)
        text_encoder_dict = {"cond_stage_model.transformer." + k for k, v in text_encoder_dict.items()}

    # Put together new checkpoint
    state_dict = {**unet_dict, **vae_dict, **text_encoder_dict}
    ckpt = {"state_dict": state_dict}

    with open(output, 'wb') as f:
        torch.save(ckpt, f)


if __name__ == '__main__':
    main()
