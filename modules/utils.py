import itertools
import logging
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Optional, Pattern, Callable, Any

import PIL.Image as Image
import lightning_utilities.core.rank_zero as rank_zero
import torch
from omegaconf import ListConfig, DictConfig

STATE_DICT = dict[str, torch.Tensor]
SUPPORTED_FORMATS = ["pt", "safetensors"]

DTYPE_MAP = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "bf16": torch.bfloat16
}


def rank_zero_info(message: str):
    if getattr(rank_zero.rank_zero_only, "rank", None) is None:
        logging.getLogger().info(message)
        return

    rank_zero.rank_zero_info(message)


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


def list_images(*paths: Path) -> Iterable[Path]:
    return itertools.chain(*((
        x for x in path.iterdir() if
        x.is_file() and
        x.suffix.lower() != ".txt" and
        x.suffix.lower() in Image.registered_extensions().keys()
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
