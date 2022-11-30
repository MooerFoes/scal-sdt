from pathlib import Path

import click
import torch

from modules.convert.common import DTYPE_CHOICES, STATE_DICT, get_module_state_dict
from modules.convert.diffusers_to_sd import convert_unet_state_dict, convert_vae_state_dict


def infer_format_from_path(path: Path):
    suffix = path.suffix[1:].lower()

    if suffix == "ckpt":
        return "pt"
    elif suffix == "safetensors":
        return "safetensors"

    return None


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
@click.option("--map-location", type=str, default="cpu",
              help='Where the checkpoint is loaded to. Could be "cpu" or "cuda".')
@click.option("--format",
              type=click.Choice(["pt", "safetensors"]),
              default=None,
              help='Save in which format. If not specified, infered from output path extension.')
def main(checkpoint, output,
         text_encoder, vae,
         unet_dtype, vae_dtype, text_encoder_dtype,
         overwrite, map_location, format):
    """Converts SCAL-SDT checkpoint to Stable Diffusion format.

    CHECKPOINT: Path to the SCAL-SDT checkpoint.
    OUTPUT: Output Stable Diffusion checkpoint path."""
    output = Path(output)

    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} already exists")

    if format is None:
        infered_format = infer_format_from_path(output)
        if infered_format is None:
            raise "Must specify a known extension or format"

        format = infered_format

    state_dict: STATE_DICT = torch.load(checkpoint, map_location=map_location)["state_dict"]

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
        text_encoder_dict = {"cond_stage_model.transformer." + k: v for k, v in text_encoder_dict.items()}

    # Put together new checkpoint
    state_dict = {**unet_dict, **vae_dict, **text_encoder_dict}

    if format == "pt":
        with open(output, 'wb') as f:
            root = {"state_dict": state_dict}
            torch.save(root, f)
    elif format == "safetensors":
        try:
            from safetensors.torch import save_file
        except ImportError:
            raise 'In order to use safetensors, run "pip install safetensors"'

        root = {"state_dict": {k: v.contiguous().to_dense() for k, v in state_dict.items()}}
        save_file(root, output)
    else:
        raise 'Invalid format'


if __name__ == '__main__':
    main()
