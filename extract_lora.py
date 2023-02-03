import logging
from math import sqrt
from pathlib import Path
from typing import Optional

import click
import torch.linalg
from omegaconf import OmegaConf
from torch import nn
from typing.io import IO

from modules import configs
from modules.configs import get_ldm_config
from modules.model import load_ldm_checkpoint, load_df_pipeline
from modules.utils import check_overwrite, SUPPORTED_FORMATS, save_state_dict, DTYPE_MAP, try_then_default, timeit, \
    apply_module_config

logger = logging.getLogger("lora-approx")


def lora_approx(delta_w: torch.Tensor, rank: int):
    """
    Apply low-rank approximation to a weight difference using SVD,
    so for input x, x @ w + x @ v_t @ u is close to x @ w + x @ delta_w.
    (v_t, u) corresponds to (lora_down, lora_up).
    """
    u, s, v_t = torch.linalg.svd(delta_w)

    u = u[:, :rank]
    s = s[:rank]

    u = u @ torch.diag(s)
    v_t = v_t[:rank, :]

    # Error = (s.norm() - delta_w.norm()).abs()

    return v_t, u


@click.command()
@click.argument("model", type=click.Path(exists=True, path_type=Path))
@click.argument("base_model", type=click.Path(exists=True, path_type=Path))
@click.argument("output", type=click.Path(path_type=Path))
@click.option("--layer-spec",
              type=click.Path(path_type=Path),
              default=configs.OPTIM_TARGETS_DIR / "lora.yaml",
              help="The layer specification, examples given at config/optim_targets.")
@click.option("--overwrite",
              is_flag=True,
              help="Allow overwriting output path if true.")
@click.option("--device",
              type=str,
              default="cpu",
              help='Tensors loading location. Possible choices are "cpu" or "cuda".')
@click.option("--dtype",
              type=click.Choice(DTYPE_MAP.keys()),
              default="fp16",
              help='Save weights in this data type.')
@click.option("--format",
              type=click.Choice(SUPPORTED_FORMATS),
              default=None,
              help='State dict saving format. If not specified, infered from output path extension.')
@torch.no_grad()
def main(model: Path,
         base_model: Path,
         output: Path,
         layer_spec: IO[str],
         overwrite: bool,
         device: str,
         dtype: str,
         format: Optional[str]):
    """
    Extract difference between a full model and its base given a layer specification, then compute a low-rank approximation using SVD.

    Save format is AddNet [1] compatible.

    If using --device cuda, SVD solving will be ~15x faster than --device cpu depending on your actual specs.

    [1] AddNet: https://github.com/kohya-ss/sd-webui-additional-networks
    """
    check_overwrite(output, overwrite)

    layer_config = OmegaConf.load(layer_spec)

    ldm_config = None

    def load_components(path: Path):
        if path.is_file():
            nonlocal ldm_config
            if ldm_config is None:
                ldm_config = get_ldm_config(configs.default().ldm_config)

            return load_ldm_checkpoint(path, ldm_config)
        else:
            return load_df_pipeline(str(path))

    unet, _, text_encoder, _, _ = load_components(model)
    unet_base, _, text_encoder_base, _, _ = load_components(base_model)

    logger.info("Weights loaded")

    unet_config, text_encoder_config = \
        try_then_default(lambda: layer_config.unet.targets), \
            try_then_default(lambda: layer_config.text_encoder.targets)

    submodules = {}

    for prefix, module, module_base, module_config in \
            [("lora_unet", unet, unet_base, unet_config),
             ("lora_te_text_model", text_encoder, text_encoder_base, text_encoder_config)]:
        if module_config is None:
            continue

        def to_kohya_format(path: str):
            return f'{prefix}_{path.replace(".", "_")}'

        def add_module(m, c, p):
            submodules[to_kohya_format(p)] = [c, m.to(device)]

        def add_base_module(m, _, p):
            submodules[to_kohya_format(p)].append(m.to(device))

        apply_module_config(module, module_config, add_module)
        apply_module_config(module_base, module_config, add_base_module)

    logger.info("Layer specification loaded and locked")

    state = {}
    svd_total_time = 0.

    for submodule_path, (submodule_config, submodule, submodule_base) in submodules.items():
        lora_config = submodule_config.get("lora")
        if lora_config is None:
            continue

        if isinstance(submodule, nn.Linear):
            (down, up), t = timeit(lambda: lora_approx(
                submodule.weight - submodule_base.weight, lora_config.rank))
        elif isinstance(submodule, nn.Conv2d) and submodule.kernel_size == (1, 1):
            (down, up), t = timeit(lambda: lora_approx(
                (submodule.weight - submodule_base.weight).squeeze(), lora_config.rank))
        else:
            raise Exception("Only Linear and Conv2d(kernel_size=(1,1)) supports LoRA.")

        svd_total_time += t

        # X @ c Vt @ c U == c^2 X @ Vt @ U
        scale = sqrt(lora_config.rank / lora_config.alpha)

        state[f"{submodule_path}.lora_down.weight"] = (down * scale).to(DTYPE_MAP[dtype])
        state[f"{submodule_path}.lora_up.weight"] = (up * scale).to(DTYPE_MAP[dtype])
        state[f"{submodule_path}.alpha"] = torch.tensor(lora_config.alpha, dtype=torch.int32)

    logger.info(f"All SVD completed, total time {svd_total_time}s")
    save_state_dict(state, output, format)


if __name__ == '__main__':
    logging.basicConfig(level="INFO")
    main()
