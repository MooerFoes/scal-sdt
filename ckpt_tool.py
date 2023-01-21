import functools
import logging
import warnings
from pathlib import Path
from typing import Optional

import click
import torch
from omegaconf import OmegaConf
from torch import nn

from modules import config
from modules.config import get_ldm_config
from modules.convert.diffusers_to_sd import convert_unet_state_dict
from modules.convert.sd_to_diffusers import convert_ldm_unet_checkpoint, create_unet_diffusers_config
from modules.model import load_components, apply_module_config
from modules.utils import check_overwrite, save_state_dict, load_state_dict, SUPPORTED_FORMATS, DTYPE_MAP, \
    try_then_default, search_key

logger = logging.getLogger("ckpt-tool")


@click.group()
def main():
    pass


@main.command()
@click.argument("checkpoint", type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path))
@click.argument("output", type=click.Path(path_type=Path))
@click.option("--text-encoder",
              is_flag=True,
              help="Whether to include text encoder weights.")
@click.option("--vae",
              type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
              help="Path to VAE ckpt.")
@click.option("--unet-dtype",
              type=click.Choice(DTYPE_MAP.keys()),
              default="fp32",
              help="Save unet weights in this data type.")
@click.option("--vae-dtype",
              type=click.Choice(DTYPE_MAP.keys()),
              default="fp32",
              help="Save VAE weights in this data type. (other than fp32 NOT RECOMMENDED)")
@click.option("--text-encoder-dtype",
              type=click.Choice(DTYPE_MAP.keys()),
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
@torch.no_grad()
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


@main.command("lora")
@click.argument("checkpoint", type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path))
@click.argument("output", type=click.Path(path_type=Path))
@click.option("--overwrite",
              is_flag=True,
              help="Whether to overwrite output.")
@click.option("--map-location",
              type=str,
              default="cpu",
              help='Where the checkpoint is loaded to. Could be "cpu" or "cuda".')
@click.option("--format",
              type=click.Choice(SUPPORTED_FORMATS),
              default=None,
              help='Save in which format. If not specified, infered from output path extension.')
@click.option("--unscale",
              is_flag=True,
              help='Scale low rank tensors by rank / alpha.')
@click.option("--dtype",
              type=click.Choice(DTYPE_MAP.keys()),
              default="fp16",
              help='Save weights in this data type.')
@torch.no_grad()
def extract_lora(checkpoint: Path,
                 output: Path,
                 overwrite: bool,
                 map_location: str,
                 format: Optional[str],
                 unscale: bool,
                 dtype: str):
    """
    Extract LoRA from a SCAL-SDT(!) checkpoint, then save to AddNet [1] compatible format.

    [1] AddNet: https://github.com/kohya-ss/sd-webui-additional-networks
    """
    check_overwrite(output, overwrite)

    def get_scale():
        if not unscale:
            return None

        run_config_path = checkpoint.parent / "config.yaml"

        if not run_config_path.exists():
            logger.warning("No corresponding config found for checkpoint, will not unscale")
            return None

        optim_target = OmegaConf.load(run_config_path).optim_target
        lora_config = next(search_key(optim_target, "lora"))
        return lora_config.alpha / lora_config.rank

    scale = get_scale()
    dtype = DTYPE_MAP[dtype]

    state_dict = load_state_dict(checkpoint, map_location)

    unet_state = convert_ldm_unet_checkpoint(state_dict,
                                             create_unet_diffusers_config(get_ldm_config(config.default().ldm_config)))

    def to_kohya_format(key: str):
        return key.replace(".", "_") \
            .replace("_lora_A", ".lora_down.weight") \
            .replace("_lora_B", ".lora_up.weight")

    def scale_lora(x: torch.Tensor):
        if scale is not None:
            x *= scale

        x = x.to(dtype)  # TODO: underflow, and upstream AddNet problem
        return x

    unet_state = {
        "lora_unet_" + to_kohya_format(k):
            scale_lora(v)
        for k, v in unet_state.items()
        if "lora" in k
    }

    text_encoder_state = {
        to_kohya_format(k).replace("cond_stage_model_transformer_text_model", "lora_te_text_model"):
            scale_lora(v)
        for k, v in state_dict.items()
        if "lora" in k and k.startswith("cond_stage_model")
    }

    state_dict = {**text_encoder_state, **unet_state}

    save_state_dict(state_dict, output, format)


def load_as_diffusers_state(path: Path, device: str, ldm_config_path: Optional[str] = None):
    diffusers = path.is_dir()

    def load_diffusers_state(module: str):
        return load_state_dict(next(n for n in (path / module).iterdir() if n.name != "config.json"), device)

    if diffusers:
        unet_state = load_diffusers_state("unet")
        clip_state = load_diffusers_state("text_encoder")
    else:
        state = load_state_dict(path, device)
        unet_state = convert_ldm_unet_checkpoint(state, create_unet_diffusers_config(get_ldm_config(ldm_config_path)))
        clip_state = {k.replace("cond_stage_model.transformer.", "text_model."): v for k, v in state.items()
                      if k.startswith("cond_stage_model.transformer.")}

    return unet_state, clip_state


@main.command("graft")
@click.argument("base_model_path", type=click.Path(exists=True, path_type=Path), nargs=1)
@click.argument("model_paths", type=click.Path(exists=True, path_type=Path), nargs=-1)
@click.argument("output_path", type=click.Path(path_type=Path), nargs=1)
@click.option("--layer-spec",
              type=click.Path(path_type=Path),
              required=True,
              help="The layer specification, examples given at config/optim_targets.")
@click.option("--overwrite",
              is_flag=True,
              help="Allow overwriting output path if true.")
@click.option("--device",
              type=str,
              default="cpu",
              help='Tensors loading location. Possible choices are "cpu" or "cuda".')
@click.option("--format",
              type=click.Choice(SUPPORTED_FORMATS),
              default=None,
              help='State dict saving format. If not specified, infered from output path extension.')
@click.option("--unet-dtype",
              type=click.Choice(DTYPE_MAP.keys()),
              default="fp32",
              help="Save unet weights in this data type.")
@click.option("--text-encoder-dtype",
              type=click.Choice(DTYPE_MAP.keys()),
              default="fp32",
              help="Save text encoder weights in data type.")
@click.option("--lru-cache-size",
              type=int,
              default=3,
              help="Max count of models stored in memory.")
@click.option("--ldm-config",
              type=str,
              default=None,
              help="Link or path to the LDM config.")
@torch.no_grad()
def graft(base_model_path: Path,
          model_paths: list[Path],
          output_path: Path,
          layer_spec: Path,
          overwrite: bool,
          device: str,
          format: str,
          unet_dtype: str,
          text_encoder_dtype: str,
          lru_cache_size: int,
          ldm_config: Optional[str]):
    """
    "Graft" one or multiple models to a base model given a layer specification.
    """
    check_overwrite(output_path, overwrite)

    layer_config = OmegaConf.load(layer_spec)

    components = load_components(str(base_model_path), ldm_config_path=ldm_config)
    base_unet, base_clip = components[0], components[2]
    del components

    unet_config, text_encoder_config = \
        try_then_default(lambda: layer_config.unet.targets), \
            try_then_default(lambda: layer_config.text_encoder.targets)

    cached_load_as_diffusers_state = functools.lru_cache(maxsize=lru_cache_size)(load_as_diffusers_state)

    for i, (module, module_config) in \
            enumerate([(base_unet, unet_config),
                       (base_clip, text_encoder_config)]):
        if module_config is None:
            continue

        def process_submodule(submodule: nn.Module, submodule_config, p):
            source_index = submodule_config.get("source")
            if source_index is None:
                return

            states = cached_load_as_diffusers_state(model_paths[source_index], device)
            submodule_state = {k.removeprefix(f"{p}."): v for k, v in states[i].items() if k.startswith(f"{p}.")}
            submodule.load_state_dict(submodule_state)

        apply_module_config(module, module_config, process_submodule)

    logger.info("Process complete wrt layer specification")

    unet_state = convert_unet_state_dict(base_unet.state_dict())
    unet_state = {"model.diffusion_model." + k: v.to(DTYPE_MAP[unet_dtype]) for k, v in unet_state.items()}

    clip_state = base_clip.state_dict()
    clip_state = {k: v.long() if v == "text_model.embeddings.position_ids" else v.to(DTYPE_MAP[text_encoder_dtype]) for
                  k, v in clip_state.items()}

    state = {**unet_state, **clip_state}

    save_state_dict(state, output_path, format)


if __name__ == '__main__':
    logging.basicConfig(level="INFO")
    main()
