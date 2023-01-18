import click
import torch

DTYPE_MAP = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "bf16": torch.bfloat16
}

DTYPE_CHOICES = click.Choice(list(DTYPE_MAP.keys()))
