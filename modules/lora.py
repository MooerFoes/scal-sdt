import loralib
from torch import nn


def get_linears(module: nn.Module):
    for name, sub in module.named_children():
        if isinstance(sub, nn.Linear):
            yield name, sub


def get_lora(module: nn.Linear | nn.Conv2d, rank=4, alpha=1, dropout=0.):
    if isinstance(module, nn.Linear):
        lora = loralib.Linear(module.in_features, module.out_features, rank, alpha, dropout)
    elif isinstance(module, nn.Conv2d):
        lora = loralib.Conv2d(module.in_channels, module.out_channels, module.kernel_size[0], rank, alpha, dropout)
    else:
        raise Exception("Unexpected module type")

    lora.weight = module.weight
    lora.bias = module.bias
    lora.lora_A.requires_grad = True
    lora.lora_B.requires_grad = True

    return lora.to(module.weight.device)
