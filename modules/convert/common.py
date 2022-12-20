from pathlib import Path
from typing import IO

import click
import torch

DTYPE_MAP = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "bf16": torch.bfloat16
}

DTYPE_CHOICES = click.Choice(list(DTYPE_MAP.keys()))

STATE_DICT = dict[str, torch.Tensor]


def load_state_dict(file: Path | IO[bytes], map_location="cpu") -> STATE_DICT:
    def load_io(io: IO[bytes]):
        return torch.load(io, map_location=map_location)

    if isinstance(file, Path):
        with file.open("rb") as f:
            ckpt = load_io(f)
    else:
        ckpt = load_io(file)

    return ckpt.get("state_dict", ckpt)
