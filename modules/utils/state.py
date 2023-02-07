from pathlib import Path
from typing import Optional

import torch

STATE = dict[str, torch.Tensor]

SUPPORTED_FORMATS = ["pt", "safetensors"]

DTYPE_MAP = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "bf16": torch.bfloat16
}


def infer_framework(state: STATE):
    if any("model.diffusion_model." in k for k in state.keys()):
        return "ldm"

    return "df"


def infer_format(path: Path):
    suffix = path.suffix[1:].lower()

    if suffix == "ckpt" or suffix == "pt":
        return "pt"
    elif suffix == "safetensors":
        return "safetensors"

    return None


def throw_if_unsupported(_format: Optional[str]):
    if _format not in SUPPORTED_FORMATS:
        raise Exception("Must specify a known extension or a format")


def save_state_dict(state: STATE, path: Path, _format: Optional[str] = None):
    if _format is None:
        _format = infer_format(path)

    throw_if_unsupported(_format)

    if _format == "pt":
        with open(path, 'wb') as f:
            torch.save(state, f)
    elif _format == "safetensors":
        state = {k: v.contiguous().to_dense() for k, v in state.items()}
        from safetensors.torch import save_file
        save_file(state, str(path))
    else:
        assert False


def load_state_dict(path: Path, device="cpu", _format: Optional[str] = None) -> STATE:
    if _format is None:
        _format = infer_format(path)

    throw_if_unsupported(_format)

    if _format == "pt":
        state = torch.load(path, device)
        state = state.get("state_dict", state)
    elif _format == "safetensors":
        from safetensors import safe_open

        state = {}
        with safe_open(path, framework="pt", device=device) as f:
            for k in f.keys():
                state[k] = f.get_tensor(k)
    else:
        assert False

    return state
