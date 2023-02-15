import contextlib
import copy

import torch
from torch import nn


# Based on torchema
class ExponentialMovingAverage:
    """
    Maintains (exponential) moving average of a set of parameters.

    Args:
        module: The module to track.

        decay: The exponential decay.

        use_num_updates: Whether to use number of updates when computing
            averages.
    """

    def __init__(
        self,
        module: nn.Module,
        decay: float,
        use_num_updates: bool = True
    ):
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.module = module
        self.shadow_params = {
            name: param.clone().detach()
            for name, param in module.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self) -> None:
        """
        Update currently maintained parameters.

        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.
        """
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(
                decay,
                (1 + self.num_updates) / (10 + self.num_updates)
            )
        one_minus_decay = 1.0 - decay

        for name, param in self.module.named_parameters():
            s_param = self.shadow_params[name]
            tmp = (s_param - param)
            # tmp will be a new tensor so we can do in-place
            tmp.mul_(one_minus_decay)
            s_param.sub_(tmp)

    def apply(self) -> None:
        """
        Copy current averaged parameters into given collection of parameters.
        """
        for name, param in self.module.named_parameters():
            s_param = self.shadow_params[name]
            param.data.copy_(s_param.data)

    @contextlib.contextmanager
    def average_parameters(self):
        """
        Context manager for validation/inference with averaged parameters.
        """
        collected_params = {
            name: param.clone()
            for name, param in self.module.named_parameters()
        }
        self.apply()
        try:
            yield
        finally:
            for name, param in self.module.named_parameters():
                param.data.copy_(collected_params[name].data)

    def to(self, device=None, dtype=None) -> None:
        r"""Move internal buffers of the ExponentialMovingAverage to `device`.

        Args:
            device: like `device` argument to `torch.Tensor.to`
        """
        # .to() on the tensors handles None correctly
        self.shadow_params = {
            name: param.to(device=device, dtype=dtype)
            if param.is_floating_point()
            else param.to(device=device)
            for name, param in self.shadow_params.items()
        }

    def state_dict(self) -> dict:
        r"""Returns the state of the ExponentialMovingAverage as a dict."""
        # Following PyTorch conventions, references to tensors are returned:
        # "returns a reference to the state and not its copy!" -
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow_params": self.shadow_params
        }

    def load_state_dict(self, state_dict: dict) -> None:
        r"""Loads the ExponentialMovingAverage state.

        Args:
            state_dict (dict): EMA state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # deepcopy, to be consistent with module API
        state_dict = copy.deepcopy(state_dict)

        decay = state_dict["decay"]
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')

        num_updates = state_dict["num_updates"]
        assert num_updates is None or isinstance(num_updates, int), \
            "Invalid num_updates"

        shadow_params = state_dict["shadow_params"]
        assert isinstance(shadow_params, dict), \
            "shadow_params must be a dict"
        assert all(
            isinstance(p, torch.Tensor) for p in shadow_params.values()
        ), "shadow_params must all be Tensors"
        assert len(self.shadow_params) == len(list(self.module.named_parameters()))

        self.decay = decay
        self.num_updates = num_updates
        self.shadow_params = shadow_params
