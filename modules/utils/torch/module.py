from types import MethodType
from typing import Callable, Optional

from omegaconf import ListConfig, DictConfig, OmegaConf
from torch import nn


def set_submodule(module: nn.Module, name: str, sub: nn.Module):
    segments = name.split(".")
    module = module.get_submodule(".".join(segments[:-1]))
    module.__setattr__(segments[-1], sub)


def apply_module_config(module: nn.Module, module_configs: ListConfig,
                        fn: Callable[[nn.Module, DictConfig, str], None], recursive=True,
                        path="", recurse_config: Optional[DictConfig] = None):
    """
    Apply a function to each submodules selected from a module w.r.t. a list of targets.

    Args:
        module: Module to begin with.
        module_configs: Configs for the module.
        fn: Function to be applied.
        recursive: Whether to apply targets recursively.
        path: Normally should not be directly given.
        recurse_config: Normally should not be directly given.

    """

    for module_config in module_configs:
        module_config: DictConfig
        index = module_config.get("index")
        targets = module_config.get("targets")

        current_depth = module_config.get("recurse_conf")
        if recurse_config is None:
            recurse_config = current_depth
        elif current_depth is not None:
            recurse_config = OmegaConf.merge(recurse_config, current_depth)

        def invoke_on_submodule(_submodule: nn.Module, _module_path: str):
            _path = _module_path if path == "" else f"{path}.{_module_path}"
            if recursive and targets is not None:
                apply_module_config(_submodule, targets, fn,
                                    path=_path, recurse_config=recurse_config)
            else:
                if recurse_config is None:
                    config = module_config
                else:
                    config = OmegaConf.merge(module_config, recurse_config)

                fn(_submodule, config, _path)

        if index is None:
            for name, submodule in module.named_children():
                if submodule == module:
                    continue

                invoke_on_submodule(submodule, name)
        else:
            for module_path in index:
                submodule = module.get_submodule(module_path)
                invoke_on_submodule(submodule, module_path)


def freeze_permanently(module: nn.Module):
    module.requires_grad_(False)
    module.eval()
    module.train = MethodType(lambda self, mode: self, module)
