import logging
from pathlib import Path
from typing import Any

import click
import torch
from tqdm import tqdm

from modules.configs import get_ldm_config
from modules.convert.sd_to_diffusers import create_unet_diffusers_config, \
    renew_resnet_paths, renew_attention_paths, assign_to_checkpoint
from modules.utils.state import save_state_dict, SUPPORTED_FORMATS, DTYPE_MAP

logger = logging.getLogger("cldm-to-diffusers")


def convert_cldm_blocks_state(state, config):
    new_checkpoint = {
        "time_embedding.linear_1.weight": state["time_embed.0.weight"],
        "time_embedding.linear_1.bias": state["time_embed.0.bias"],
        "time_embedding.linear_2.weight": state["time_embed.2.weight"],
        "time_embedding.linear_2.bias": state["time_embed.2.bias"],
        "conv_in.weight": state["input_blocks.0.0.weight"],
        "conv_in.bias": state["input_blocks.0.0.bias"],
    }

    # Retrieves the keys for the input blocks only
    num_input_blocks = len({".".join(layer.split(".")[:2]) for layer in state if "input_blocks" in layer})
    input_blocks = {
        layer_id: [key for key in state if f"input_blocks.{layer_id}" in key]
        for layer_id in range(num_input_blocks)
    }

    # Retrieves the keys for the middle blocks only
    num_middle_blocks = len({".".join(layer.split(".")[:2]) for layer in state if "middle_block" in layer})
    middle_blocks = {
        layer_id: [key for key in state if f"middle_block.{layer_id}" in key]
        for layer_id in range(num_middle_blocks)
    }

    for i in range(1, num_input_blocks):
        block_id = (i - 1) // (config["layers_per_block"] + 1)
        layer_in_block_id = (i - 1) % (config["layers_per_block"] + 1)

        resnets = [
            key for key in input_blocks[i] if f"input_blocks.{i}.0" in key and f"input_blocks.{i}.0.op" not in key
        ]
        attentions = [key for key in input_blocks[i] if f"input_blocks.{i}.1" in key]

        if f"input_blocks.{i}.0.op.weight" in state:
            new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.weight"] = state.pop(
                f"input_blocks.{i}.0.op.weight"
            )
            new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.bias"] = state.pop(
                f"input_blocks.{i}.0.op.bias"
            )

        paths = renew_resnet_paths(resnets)
        meta_path = {"old": f"input_blocks.{i}.0", "new": f"down_blocks.{block_id}.resnets.{layer_in_block_id}"}
        assign_to_checkpoint(
            paths, new_checkpoint, state, additional_replacements=[meta_path], config=config
        )

        if len(attentions):
            paths = renew_attention_paths(attentions)
            meta_path = {"old": f"input_blocks.{i}.1", "new": f"down_blocks.{block_id}.attentions.{layer_in_block_id}"}
            assign_to_checkpoint(
                paths, new_checkpoint, state, additional_replacements=[meta_path], config=config
            )

    resnet_0 = middle_blocks[0]
    attentions = middle_blocks[1]
    resnet_1 = middle_blocks[2]

    resnet_0_paths = renew_resnet_paths(resnet_0)
    assign_to_checkpoint(resnet_0_paths, new_checkpoint, state, config=config)

    resnet_1_paths = renew_resnet_paths(resnet_1)
    assign_to_checkpoint(resnet_1_paths, new_checkpoint, state, config=config)

    attentions_paths = renew_attention_paths(attentions)
    meta_path = {"old": "middle_block.1", "new": "mid_block.attentions.0"}
    assign_to_checkpoint(
        attentions_paths, new_checkpoint, state, additional_replacements=[meta_path], config=config
    )

    return new_checkpoint


def replace_prefix(d: dict[str, Any], prefix="", replacement=""):
    return {replacement + k[len(prefix):]: v for k, v in d.items() if k.startswith(prefix)}


def convert_controlnet_state(state, config):
    state = replace_prefix(state, "control_model.")

    conv_in = replace_prefix(state, "input_hint_block.", "cond_image_conv_in.")

    projections = {}

    for i in range(12):
        for p in ["weight", "bias"]:
            projections[f"down_connect_projections.{i}.{p}"] = state.pop(f"zero_convs.{i}.0.{p}")

    for p in ["weight", "bias"]:
        projections[f"mid_connect_projection.{p}"] = state.pop(f"middle_block_out.0.{p}")

    others = convert_cldm_blocks_state(state, config)

    return {**conv_in, **projections, **others}


@click.command()
@click.argument("inputs",
                nargs=-1,
                type=click.Path(exists=True, file_okay=True, path_type=Path))
@click.argument("output_dir",
                nargs=1,
                type=click.Path(path_type=Path))
@click.option("--format",
              type=click.Choice(SUPPORTED_FORMATS),
              default="safetensors",
              help='Save in which format.')
@click.option("--ldm-config",
              type=str,
              default=None,
              help="Link or path to the LDM config.")
@click.option("--map-location",
              type=str,
              default="cpu",
              help='Where the checkpoint is loaded to. Could be "cpu" or "cuda".')
@click.option("--dtype",
              type=click.Choice(DTYPE_MAP.keys()),
              default="fp16",
              help='Save weights in this data type.')
@torch.no_grad()
def main(inputs: list[Path], output_dir: Path, format: str, ldm_config: str, map_location: str, dtype: str):
    logging.warning("This script does not guarantee the same result as original repo")

    output_dir.mkdir(exist_ok=True)

    ldm_config = get_ldm_config(ldm_config)
    config = create_unet_diffusers_config(ldm_config)

    dtype = DTYPE_MAP[dtype]

    for path in tqdm(inputs):
        original_state = torch.load(path, map_location)

        if any("output_blocks" in k for k in original_state.keys()):
            logger.warning(f"{path.name}: Will drop all up block states")

        state = convert_controlnet_state(original_state, config)
        state = {k: v.to(dtype) for k, v in state.items()}

        save_state_dict(state, output_dir / f"{path.stem}.safetensors", format)


if __name__ == "__main__":
    main()
    logging.basicConfig()
