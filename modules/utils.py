import itertools
import logging
import time
from collections.abc import Iterable
from pathlib import Path
from types import MethodType
from typing import Optional, Pattern, Callable, Any

import PIL.Image as Image
import lightning_utilities.core.rank_zero as rank_zero
import torch
from omegaconf import ListConfig, DictConfig, OmegaConf
from torch import nn

STATE_DICT = dict[str, torch.Tensor]
SUPPORTED_FORMATS = ["pt", "safetensors"]

DTYPE_MAP = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "bf16": torch.bfloat16
}


def rank_zero_logger(name: Optional[str] = None):
    logger = logging.getLogger(name)

    if getattr(rank_zero, "rank", None) is None:
        return logger

    log_rank_zero = rank_zero.rank_zero_only(logger._log)
    logger._log = MethodType(log_rank_zero, logger)
    return logger


def get_string(link_or_path: str):
    if link_or_path.startswith("http://") or link_or_path.startswith("https://"):
        import requests
        with requests.Session() as session:
            content = session.get(link_or_path).content.decode("utf-8")
    elif Path(link_or_path).exists():
        with open(link_or_path, "r") as f:
            content = f.read()
    else:
        raise ValueError(f'"{link_or_path}" is not a valid link or path')

    return content


def infer_model_from_state_dict(state: STATE_DICT):
    if any("model.diffusion_model." in k for k in state.keys()):
        return "ldm"

    return "df"


def infer_format_from_path(path: Path):
    suffix = path.suffix[1:].lower()

    if suffix == "ckpt" or suffix == "pt":
        return "pt"
    elif suffix == "safetensors":
        return "safetensors"

    return None


def check_overwrite(path: Path, overwrite: bool):
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists")


def save_state_dict(state: STATE_DICT, path: Path, format: Optional[str] = None):
    if format is None:
        format = infer_format_from_path(path)

    if format == "pt":
        with open(path, 'wb') as f:
            torch.save(state, f)
    elif format == "safetensors":
        state = {k: v.contiguous().to_dense() for k, v in state.items()}
        from safetensors.torch import save_file
        save_file(state, str(path))
    else:
        raise Exception("Must specify a known extension or a format.")


def load_state_dict(path: Path, device="cpu", format: Optional[str] = None,
                    target_keys_regex: Optional[Pattern] = None) -> STATE_DICT:
    if format is None:
        format = infer_format_from_path(path)

    if format == "pt":
        state = torch.load(path, device)
        state = state.get("state_dict", state)
        if target_keys_regex is not None:
            state = {k: v for k, v in state.items() if target_keys_regex.match(k) is not None}
    elif format == "safetensors":
        from safetensors import safe_open

        state = {}
        with safe_open(path, framework="pt", device=device) as f:
            for k in f.keys():
                state[k] = f.get_tensor(k)
    else:
        raise Exception("Must specify a known extension or a format.")

    return state


# Hardcode an extension set, due to PIL.Image.registered_extensions() returns too many weird stuffs.
SUPPORTED_EXTENSIONS = {'.jpe', '.jpg', '.jpeg', '.gif', '.apng', '.jfif', '.tif', '.tiff', '.bmp', '.png', '.webp'}


def is_image_file(path: Path):
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS


def list_images(*paths: Path) -> Iterable[Path]:
    return itertools.chain(*((
        x for x in path.iterdir() if is_image_file(x)
    ) for path in paths))


def read_image(filepath: Path) -> Image.Image:
    img = Image.open(filepath)

    if not img.mode == "RGB":
        img = img.convert("RGB")
    return img


def get_class(name: str):
    import importlib
    module_name, class_name = name.rsplit(".", 1)
    module = importlib.import_module(module_name, package=None)
    return getattr(module, class_name)


def physical_core_count():
    import psutil
    return psutil.cpu_count(logical=False)


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


def enumerate_dict_config(conf: ListConfig, recurse=True):
    for item in conf:
        if isinstance(item, DictConfig):
            yield item
        elif recurse:
            assert isinstance(conf, ListConfig)
            yield from enumerate_dict_config(item)


def search_key(conf: ListConfig | DictConfig, key: str, recurse=True):
    if isinstance(conf, DictConfig):
        value = conf.get(key)
        if value is not None:
            yield value

        if not recurse:
            return

        for item in conf.values():
            if not (isinstance(item, ListConfig) or isinstance(item, DictConfig)):
                continue

            yield from search_key(item, key, True)
    elif recurse:
        assert isinstance(conf, ListConfig)
        for item in enumerate_dict_config(conf, False):
            yield from search_key(item, key, True)


class TransformersNoStupidWarnings:
    """Silent warnings like "Some weights of the model checkpoint at openai/clip-vit-large-patch14 were not used"."""
    from transformers import logging
    def __enter__(self):
        self._prev_verbosity = self.logging.get_verbosity()
        self.logging.set_verbosity_error()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logging.set_verbosity(self._prev_verbosity)


def set_submodule(module: nn.Module, name: str, sub: nn.Module):
    segments = name.split(".")
    module = module.get_submodule(".".join(segments[:-1]))
    module.__setattr__(segments[-1], sub)


def apply_module_config(module: nn.Module, module_configs: ListConfig,
                        fn: Callable[[nn.Module, DictConfig, str], None], recursive=True,
                        path="", recurse_config: Optional[DictConfig] = None):
    """
    Apply a function to each submodules selected from a module w.r.t. a list of targets.

    Args:
        module: Module to begin with.
        module_configs: Configs for the module.
        fn: Function to be applied.
        recursive: Whether to apply targets recursively.
        path: Normally should not be directly given.
        recurse_config: Normally should not be directly given.

    """

    for module_config in module_configs:
        module_config: DictConfig
        index = module_config.get("index")
        targets = module_config.get("targets")

        current_depth = module_config.get("recurse_conf")
        if recurse_config is None:
            recurse_config = current_depth
        elif current_depth is not None:
            recurse_config = OmegaConf.merge(recurse_config, current_depth)

        def invoke_on_submodule(_submodule: nn.Module, _module_path: str):
            _path = _module_path if path == "" else f"{path}.{_module_path}"
            if recursive and targets is not None:
                apply_module_config(_submodule, targets, fn,
                                    path=_path, recurse_config=recurse_config)
            else:
                if recurse_config is None:
                    config = module_config
                else:
                    config = OmegaConf.merge(module_config, recurse_config)

                fn(_submodule, config, _path)

        if index is None:
            for name, submodule in module.named_children():
                if submodule == module:
                    continue

                invoke_on_submodule(submodule, name)
        else:
            for module_path in index:
                submodule = module.get_submodule(module_path)
                invoke_on_submodule(submodule, module_path)


def raise_if_nan(x: torch.Tensor, name: str):
    if not torch.any(torch.isnan(x)):
        return

    raise Exception(f"NaN element discovered in {name}")
