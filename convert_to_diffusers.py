from pathlib import Path
from typing import Optional

import click
from diffusers import StableDiffusionPipeline

from modules.config import get_ldm_config
from modules.model import load_ldm_checkpoint
from modules.utils import DTYPE_MAP


@click.command()
@click.argument("checkpoint", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output", type=click.Path(path_type=Path))
@click.option("--config",
              type=str,
              default=None,
              help="Link or path to the LDM config.")
@click.option("--unet-dtype",
              type=click.Choice(DTYPE_MAP.keys()),
              default="fp32",
              help="Save unet weights in this data type.")
@click.option("--vae",
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Path to VAE ckpt.")
# @click.option("--vae-dtype",
#               type=DTYPE_CHOICES,
#               default="fp32",
#               help="Save VAE weights in this data type. (other than fp32 NOT RECOMMENDED)")
# @click.option("--text-encoder-dtype",
#               type=DTYPE_CHOICES,
#               default="fp32",
#               help="Save text encoder weights in data type.")
@click.option("--overwrite", is_flag=True)
def main(checkpoint: Path,
         output: Path,
         ldm_config_path: Optional[str],
         unet_dtype: str,
         vae: Optional[Path],
         # vae_dtype: str,
         # text_encoder_dtype: str,
         overwrite: bool):
    """Converts SCAL-SDT checkpoint to Diffusers saved pipeline.

    CHECKPOINT: Path to the SCAL-SDT checkpoint.
    OUTPUT: Diffusers pipeline output path."""
    if output.exists() and not overwrite:
        raise FileExistsError(f'"{output}" already exists')

    ldm_config = get_ldm_config(ldm_config_path)

    unet, vae, text_encoder, tokenizer, scheduler = load_ldm_checkpoint(checkpoint, ldm_config, vae)

    unet.to(DTYPE_MAP[unet_dtype])
    # vae.to(DTYPE_MAP[vae_dtype])
    # text_encoder.to(DTYPE_MAP[text_encoder_dtype])

    pipeline = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False
    )

    pipeline.save_pretrained(output)


if __name__ == '__main__':
    main()
