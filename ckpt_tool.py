import warnings
from pathlib import Path
from typing import Optional

import click

from modules import config
from modules.convert.common import DTYPE_CHOICES, DTYPE_MAP
from modules.convert.sd_to_diffusers import convert_ldm_unet_checkpoint, create_unet_diffusers_config
from modules.model import get_ldm_config
from modules.utils import check_overwrite, save_state_dict, load_state_dict, SUPPORTED_FORMATS


@click.group()
def cli():
    pass


@cli.command()
@click.argument("checkpoint", type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path))
@click.argument("output", type=click.Path(path_type=Path))
@click.option("--text-encoder",
              is_flag=True,
              help="Whether to include text encoder weights.")
@click.option("--vae",
              type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
              help="Path to VAE ckpt.")
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
@click.option("--overwrite",
              is_flag=True,
              help="Whether to overwrite output")
@click.option("--map-location",
              type=str,
              default="cpu",
              help='Where the checkpoint is loaded to. Could be "cpu" or "cuda".')
@click.option("--format",
              type=click.Choice(SUPPORTED_FORMATS),
              default=None,
              help='Save in which format. If not specified, infered from output path extension.')
@click.option("--ema",
              is_flag=True,
              help="Use EMA weights.")
def prune(checkpoint: Path,
          output: Path,
          text_encoder: bool,
          vae: Optional[Path],
          unet_dtype: str,
          vae_dtype: str,
          text_encoder_dtype: str,
          overwrite: bool,
          map_location: str,
          format: Optional[str],
          ema: bool):
    """Prune a SCAL-SDT checkpoint.

    CHECKPOINT: Path to the SCAL-SDT checkpoint.
    OUTPUT: Output path."""
    check_overwrite(output, overwrite)

    state_dict = load_state_dict(checkpoint, map_location)

    if not ema:
        unet_dict = {k: v.to(DTYPE_MAP[unet_dtype])
                     for k, v in state_dict.items() if k.startswith("model.diffusion_model.")}
    else:
        unet_dict = {k: v.to(DTYPE_MAP[unet_dtype])
                     for k, v in state_dict["unet_ema"]["state_dict"].items()}

    vae_dict = {}
    if vae is not None:
        vae_dict = load_state_dict(vae, map_location)
        vae_dict = {k: v.to(DTYPE_MAP[vae_dtype])
                    for k, v in vae_dict.items() if k.startswith("first_stage_model.")}

    text_encoder_dict = {}
    if text_encoder:
        text_encoder_dict = {k: v.to(DTYPE_MAP[text_encoder_dtype])
                             for k, v in state_dict.items() if k.startswith("cond_stage_model.transformer.")}
        if not any(text_encoder_dict.items()):
            warnings.warn("No text encoder weights were found in the checkpoint.")

    # Put together new checkpoint
    state_dict = {**unet_dict, **vae_dict, **text_encoder_dict}

    save_state_dict(state_dict, output, format)


@cli.command("lora")
@click.argument("checkpoint", type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path))
@click.argument("output", type=click.Path(path_type=Path))
@click.option("--overwrite",
              is_flag=True,
              help="Whether to overwrite output")
@click.option("--map-location",
              type=str,
              default="cpu",
              help='Where the checkpoint is loaded to. Could be "cpu" or "cuda".')
@click.option("--format",
              type=click.Choice(SUPPORTED_FORMATS),
              default=None,
              help='Save in which format. If not specified, infered from output path extension.')
def extract_lora(checkpoint: Path,
                 output: Path,
                 overwrite: bool,
                 map_location: str,
                 format: Optional[str]):
    check_overwrite(output, overwrite)

    state_dict = load_state_dict(checkpoint, map_location)

    unet_state = convert_ldm_unet_checkpoint(state_dict,
                                             create_unet_diffusers_config(get_ldm_config(config.default().ldm_config)))

    def to_kohya_format(key: str):
        return key.replace(".", "_") \
            .replace("_lora_A", ".lora_down.weight") \
            .replace("_lora_B", ".lora_up.weight")

    unet_state = {
        "lora_unet_" + to_kohya_format(k): v
        for k, v in unet_state.items() if "lora" in k
    }

    text_encoder_state = {
        to_kohya_format(k).replace("cond_stage_model_transformer_text_model", "lora_te_text_model"): v
        for k, v in state_dict.items() if "lora" in k and k.startswith("cond_stage_model")
    }

    state_dict = {**text_encoder_state, **unet_state}

    save_state_dict(state_dict, output, format)


if __name__ == '__main__':
    cli()
