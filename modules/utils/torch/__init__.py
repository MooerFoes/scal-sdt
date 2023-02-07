import torch


def raise_if_nan(x: torch.Tensor, name: str):
    if not torch.any(torch.isnan(x)):
        return

    raise Exception(f"NaN element discovered in {name}")
