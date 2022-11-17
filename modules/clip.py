from typing import Union, Optional, Tuple

import torch
from torch import nn
from transformers import CLIPTextModel
from transformers.modeling_outputs import BaseModelOutputWithPooling


class CLIPWithSkip(CLIPTextModel):
    skip = 1

    def _reorder_cache(self, past, beam_idx):
        super()._reorder_cache(past, beam_idx)

    def get_position_embeddings(self) -> Union[nn.Embedding, Tuple[nn.Embedding]]:
        return super().get_position_embeddings()

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        super().resize_position_embeddings(new_num_position_embeddings)

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        if self.skip > 1:
            result: BaseModelOutputWithPooling = super().forward(input_ids, attention_mask, position_ids,
                                                                 output_attentions, output_hidden_states=True,
                                                                 return_dict=True)
            hidden_state = result.hidden_states[-self.skip]
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
        else:
            return super().forward(input_ids, attention_mask, position_ids, output_attentions, output_hidden_states,
                                   return_dict)
