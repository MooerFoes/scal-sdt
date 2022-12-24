from os import PathLike
from typing import IO, Any

import click
import torch

DTYPE_MAP = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "bf16": torch.bfloat16
}

DTYPE_CHOICES = click.Choice(list(DTYPE_MAP.keys()))

STATE_DICT = dict[str, Any]


def load_state_dict(file: str | PathLike | IO[bytes], map_location="cpu") -> STATE_DICT:
    ckpt = torch.load(file, map_location=map_location)
    return ckpt.get("state_dict", ckpt)
