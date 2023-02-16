import functools
import logging
from pathlib import Path
from typing import Optional

import click
import torch
from diffusers import AutoencoderKL
from omegaconf import OmegaConf
from torch import nn
from transformers import CLIPTextModel

from modules.configs import get_ldm_config
from modules.convert.diffusers_to_sd import convert_unet_state_dict, convert_vae_state_dict
from modules.convert.sd_to_diffusers import convert_ldm_unet_checkpoint, create_unet_diffusers_config
from modules.model import load_components
from modules.text_encoders import CLIP_L
from modules.utils.config import search_key
from modules.utils.hof import try_then_default
from modules.utils.io import check_overwrite
from modules.utils.state import save_state_dict, load_state_dict, STATE, SUPPORTED_FORMATS, DTYPE_MAP, replace_prefix, \
    cast_type
from modules.utils.torch.module import apply_module_config

logger = logging.getLogger("ckpt-tool")


@click.group()
def main():
    pass


@main.command()
@click.argument("checkpoint", type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path))
@click.argument("output", type=click.Path(path_type=Path))
@click.option("--unet-dtype",
              type=click.Choice(DTYPE_MAP.keys()),
              default="fp16",
              help="Save UNet weights in this data type.")
@click.option("--text-encoder",
              is_flag=True,
              help="Include text encoder weights.")
@click.option("--text-encoder-dtype",
              type=click.Choice(DTYPE_MAP.keys()),
              default="fp16",
              help="Save text encoder weights in data type.")
@click.option("--vae",
              type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
              help="Include VAE weights. Path to a original LDM VAE or a checkpoint containing VAE.")
@click.option("--df-vae",
              type=str,
              help="Include VAE weights. Name or path to a Diffusers VAE.")
@click.option("--vae-dtype",
              type=click.Choice(DTYPE_MAP.keys()),
              default="fp32",
              help="Save VAE weights in this data type. (other than fp32 NOT RECOMMENDED)")
@click.option("--overwrite",
              is_flag=True,
              help="Overwrite existing output.")
@click.option("--map-location",
              type=str,
              default="cpu",
              help='Tensors loading location. Example: "cpu", "cuda".')
@click.option("--format",
              type=click.Choice(SUPPORTED_FORMATS),
              default=None,
              help="States saving format. If not specified, infered from output path extension.")
@click.option("--ema",
              is_flag=True,
              help="Use EMA weights.")
def prune(checkpoint: Path,
          output: Path,
          unet_dtype: str,
          text_encoder: bool,
          text_encoder_dtype: str,
          vae: Optional[Path],
          df_vae: Optional[str],
          vae_dtype: str,
          overwrite: bool,
          map_location: str,
          format: Optional[str],
          ema: bool):
    """Convert a SCAL-SDT checkpoint for use on CompVis/StabilityAI LDM codebase.

    CHECKPOINT: Path to the SCAL-SDT checkpoint.
    OUTPUT: Output path."""
    check_overwrite(output, overwrite)
    assert not (vae and df_vae), "Only one of --vae or --df-vae should be specified"

    ssdt_state = load_state_dict(checkpoint, map_location)
    ldm_state = {}

    # region UNet
    if ema:
        unet_state = ssdt_state["unet_ema"]["shadow_params"]
    else:
        unet_state = replace_prefix(ssdt_state, "unet.")

    unet_state = convert_unet_state_dict(unet_state)
    unet_state = replace_prefix(unet_state, "", "model.diffusion_model.")
    unet_state = cast_type(unet_state, unet_dtype)
    ldm_state.update(unet_state)
    # endregion

    # region VAE
    vae_state = None

    if vae is not None:
        vae_state = load_state_dict(vae, map_location)
        vae_state_from_ldm = replace_prefix(vae_state, "first_stage_model.")
        if any(vae_state_from_ldm.items()):
            vae_state = vae_state_from_ldm
        else:
            vae_state = replace_prefix(vae_state, "", "first_stage_model.")
    elif df_vae is not None:
        vae_state = AutoencoderKL.from_pretrained(df_vae).state_dict()
        vae_state = convert_vae_state_dict(vae_state)
        vae_state = replace_prefix(vae_state, "", "first_stage_model.")

    if vae_state is not None:
        vae_state = cast_type(vae_state, vae_dtype)
        ldm_state.update(vae_state)
    # endregion

    # region TE
    if text_encoder:
        te_state = replace_prefix(ssdt_state, "condition_model.encoder.text_model.", "cond_stage_model.transformer.")
        if not any(te_state.items()):
            te_state = CLIPTextModel.from_pretrained(CLIP_L).state_dict()
            te_state = replace_prefix(te_state, "text_model.", "cond_stage_model.transformer.")
        te_state = cast_type(te_state, text_encoder_dtype)
        ldm_state.update(te_state)
    # endregion

    save_state_dict(ldm_state, output, format)


@main.command("lora")
@click.argument("checkpoint", type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path))
@click.argument("output", type=click.Path(path_type=Path))
@click.option("--overwrite",
              is_flag=True,
              help="Overwrite existing output.")
@click.option("--map-location",
              type=str,
              default="cpu",
              help='Tensors loading location. Example: "cpu", "cuda".')
@click.option("--format",
              type=click.Choice(SUPPORTED_FORMATS),
              default=None,
              help="States saving format. If not specified, infered from output path extension.")
@click.option("--dtype",
              type=click.Choice(DTYPE_MAP.keys()),
              default="fp16",
              help="Save weights in this data type.")
def extract_lora(checkpoint: Path,
                 output: Path,
                 overwrite: bool,
                 map_location: str,
                 format: Optional[str],
                 dtype: str):
    """
    Extract LoRA from a SCAL-SDT(!) checkpoint, then save to AddNet [1] compatible format.

    [1] AddNet: https://github.com/kohya-ss/sd-webui-additional-networks
    """
    check_overwrite(output, overwrite)

    @functools.cache
    def get_alpha():
        run_config_path = checkpoint.parent / "config.yaml"

        if not run_config_path.exists():
            logger.warning("No corresponding config found for checkpoint, alpha will not work")
            return None

        optim_target = OmegaConf.load(run_config_path).optim_target
        lora_config = next(search_key(optim_target, "lora"))
        return lora_config.alpha

    dtype = DTYPE_MAP[dtype]

    ssdt_state = load_state_dict(checkpoint, map_location)

    def to_kohya_format(state: STATE, prefix: str):
        lora_modules = set()
        result = {}

        for k, v in state.items():
            components = k.split(".")
            if components[-1] in ["lora_A", "lora_B"]:
                lora_modules.add(".".join(components[:-1]))

        for lora_module in lora_modules:
            alpha_key = f"{lora_module}.lora_alpha"
            if alpha_key not in state:
                alpha = get_alpha()
                if alpha is not None:
                    state[alpha_key] = torch.tensor(alpha, dtype=torch.int32)

            for k, v in state.items():
                if not k.startswith(lora_module):
                    continue

                components = k.split(".")
                components, lora_key = components[:-1], components[-1]

                lora_key_map = {"lora_A": "lora_down.weight", "lora_B": "lora_up.weight", "lora_alpha": "alpha"}
                lora_key = lora_key_map.get(lora_key, None)

                if lora_key is None:
                    continue

                components.insert(0, prefix)

                k = "_".join(components) + f".{lora_key}"
                if v.dtype.is_floating_point:
                    v = v.to(dtype)

                result[k] = v

        return result

    lora_state = {}

    unet_state = replace_prefix(ssdt_state, "unet.")
    unet_state = to_kohya_format(unet_state, "lora_unet")
    lora_state.update(unet_state)

    text_encoder_state = replace_prefix(ssdt_state, "condition_model.encoder.")
    text_encoder_state = to_kohya_format(text_encoder_state, "lora_te")
    lora_state.update(text_encoder_state)

    save_state_dict(lora_state, output, format)


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
        clip_state = replace_prefix(state, "cond_stage_model.transformer.", "text_model.")

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
              help="Overwrite existing output.")
@click.option("--map-location",
              type=str,
              default="cpu",
              help='Tensors loading location. Example: "cpu", "cuda".')
@click.option("--format",
              type=click.Choice(SUPPORTED_FORMATS),
              default=None,
              help="States saving format. If not specified, infered from output path extension.")
@click.option("--unet-dtype",
              type=click.Choice(DTYPE_MAP.keys()),
              default="fp32",
              help="Save UNet weights in this data type.")
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
def graft(base_model_path: Path,
          model_paths: list[Path],
          output_path: Path,
          layer_spec: Path,
          overwrite: bool,
          map_location: str,
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

    components = load_components(str(base_model_path), ldm_config=ldm_config)
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

            states = cached_load_as_diffusers_state(model_paths[source_index], map_location)
            submodule_state = {k.removeprefix(f"{p}."): v for k, v in states[i].items() if k.startswith(f"{p}.")}
            submodule.load_state_dict(submodule_state)

        apply_module_config(module, module_config, process_submodule)

    logger.info("Process complete wrt layer specification")

    ldm_state = {}

    unet_state = convert_unet_state_dict(base_unet.state_dict())
    unet_state = replace_prefix(unet_state, "", "model.diffusion_model.")
    unet_state = cast_type(unet_state, unet_dtype)
    ldm_state.update(unet_state)

    clip_state = base_clip.state_dict()
    clip_state = cast_type(clip_state, text_encoder_dtype)
    ldm_state.update(clip_state)

    save_state_dict(ldm_state, output_path, format)


if __name__ == '__main__':
    logging.basicConfig(level="INFO")
    torch.set_grad_enabled(False)
    main()
