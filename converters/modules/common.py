import click
import torch

DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32
}

DTYPE_CHOICES = click.Choice(list(DTYPE_MAP.keys()))

STATE_DICT = dict[str, torch.Tensor]


def get_module_state_dict(state_dict: STATE_DICT, module: str, dtype) -> STATE_DICT:
    prefix = f"{module}."
    return {k.removeprefix(prefix): v.to(dtype) for k, v in state_dict.items() if k.startswith(prefix)}
