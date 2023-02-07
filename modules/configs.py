import functools
from os import PathLike
from pathlib import Path
from typing import IO, Optional

from omegaconf import OmegaConf, DictConfig

from .utils.io.net import get_string

CONFIGS_DIR = Path(__file__).parent.parent / "configs"
OPTIM_TARGETS_DIR = CONFIGS_DIR / "optim_targets"
DEFAULT_PATH = CONFIGS_DIR / '__reserved_default__.yaml'


@functools.cache
def default() -> DictConfig:
    return OmegaConf.load(DEFAULT_PATH)


@functools.cache
def get_ldm_config(link_or_path: Optional[str]) -> DictConfig:
    if link_or_path is None:
        link_or_path = default().ldm_config

    return OmegaConf.create(get_string(link_or_path))


def load_with_defaults(config: str | PathLike | IO):
    return OmegaConf.merge(default(), OmegaConf.load(config))
