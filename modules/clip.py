from typing import Union, Optional, Tuple

import torch
from transformers import CLIPTextModel
from transformers.modeling_outputs import BaseModelOutputWithPooling


def hook_forward(clip: CLIPTextModel, stop_at_layer: int):
    if stop_at_layer == -1:
        return

    original_forward = clip.forward

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        result: BaseModelOutputWithPooling = original_forward(input_ids, attention_mask, position_ids,
                                                              output_attentions, output_hidden_states=True,
                                                              return_dict=True)
        hidden_state = result.hidden_states[stop_at_layer]
        hidden_state = self.text_model.final_layer_norm(hidden_state)
        pooled_output = hidden_state[torch.arange(hidden_state.shape[0]), input_ids.argmax(dim=-1)]

        if not return_dict:
            return hidden_state, pooled_output

        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_state,
            pooler_output=pooled_output,
            hidden_states=result.hidden_states,
            attentions=result.attentions,
        )

    import types
    clip.forward = types.MethodType(forward, clip)
    return clip
