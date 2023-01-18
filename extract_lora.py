import logging
import time
from pathlib import Path
from typing import Optional, Callable, Any

import click
import torch.linalg
from omegaconf import OmegaConf
from torch import nn
from typing.io import IO

from modules import config
from modules.model import get_ldm_config, load_ldm_checkpoint, load_df_pipeline, apply_module_config
from modules.utils import check_overwrite, SUPPORTED_FORMATS, save_state_dict

logger = logging.getLogger("lora-approx")


def try_then_default(f: Callable[[], Any], default=None):
    try:
        return f()
    except:
        return default


def timeit(f: Callable[[], Any]):
    start = time.perf_counter()
    result = f()
    t = time.perf_counter() - start
    return result, t


def lora_approx(delta_w: torch.Tensor, rank: int):
    """
    Apply low-rank approximation to a dense layer weight difference using SVD,
    so for input x, x @ w + x @ u @ v_t is close to x @ w + x @ delta_w.
    (u, v_t) corresponds to (lora_down, lora_up).
    """
    u, s, v_t = torch.linalg.svd(delta_w)

    u = u[:, :rank]
    s = s[:rank]
    u = u @ torch.diag(s)
    v_t = v_t[:rank, :]

    return u, v_t


@click.command()
@click.argument("model", type=click.Path(exists=True, path_type=Path))
@click.argument("base_model", type=click.Path(exists=True, path_type=Path))
@click.argument("output", type=click.Path(path_type=Path))
@click.option("--layer-spec",
              type=click.Path(path_type=Path),
              default=config.OPTIM_TARGETS_DIR / "lora.yaml",
              help="The layer specification, examples given at config/optim_targets.")
@click.option("--overwrite",
              is_flag=True,
              help="Allow overwriting output path if true.")
@click.option("--device",
              type=str,
              default="cpu",
              help='Tensors loading location. Possible choices are "cpu" or "cuda".')
@click.option("--format",
              type=click.Choice([SUPPORTED_FORMATS]),
              default=None,
              help='State dict saving format. If not specified, infered from output path extension.')
@torch.no_grad()
def main(model: Path,
         base_model: Path,
         output: Path,
         layer_spec: IO[str],
         overwrite: bool,
         device: str,
         format: Optional[str]):
    """
    Extract difference between a full model and its base given a layer specification, then compute a low-rank approximation using SVD.

    If using --device cuda, SVD solving will be ~15x faster than --device cpu depending on your actual specs.
    """
    check_overwrite(output, overwrite)

    layer_config = OmegaConf.load(layer_spec)

    ldm_config = None

    def load_components(path: Path):
        if path.suffix.lower() == ".ckpt":
            nonlocal ldm_config
            if ldm_config is None:
                ldm_config = get_ldm_config(config.default().ldm_config)

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
        apply_module_config(module, module_config, add_base_module)

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
        elif isinstance(submodule, nn.Conv2d):
            (down, up), t = timeit(lambda: lora_approx(
                (submodule.weight - submodule_base.weight).squeeze(), lora_config.rank))
            down = down.unsqueeze(2).unsqueeze(3)
            up = up.unsqueeze(2).unsqueeze(3)
        else:
            raise Exception("Only Linear and Conv2d supports LoRA.")

        svd_total_time += t

        down *= lora_config.alpha / lora_config.rank
        up *= lora_config.alpha / lora_config.rank

        state[f"{submodule_path}.lora_down.weight"] = down
        state[f"{submodule_path}.lora_up.weight"] = up

    logger.info(f"All SVD completed, total time {svd_total_time}s")

    save_state_dict(state, output, format, overwrite)


if __name__ == '__main__':
    logging.basicConfig(level="INFO")
    main()
