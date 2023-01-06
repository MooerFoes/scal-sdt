from os import PathLike
from pathlib import Path
from typing import IO

from omegaconf import OmegaConf

CONFIGS_DIR = Path(__file__).parent.parent / "configs"
DEFAULT = CONFIGS_DIR / '__reserved_default__.yaml'


def default():
    return OmegaConf.load(DEFAULT)


def load_with_defaults(config: str | PathLike | IO):
    return OmegaConf.merge(default(), OmegaConf.load(config))
