import re
import types
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Optional

import torch
from einops import rearrange
from torch import nn
from transformers import PreTrainedTokenizerBase

from .utils import TransformersNoStupidWarnings
from .utils.logging import rank_zero_logger

CLIP_L = "openai/clip-vit-large-patch14"
CLIP_H = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
T5_L = "google/flan-t5-large"


class TextEncoder(nn.Module):
    def __init__(self, encoder, tokenizer: PreTrainedTokenizerBase, encoder_dim: int,
                 projection_dim: Optional[int] = None, max_length: Optional[int] = None):
        super().__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.max_length = max_length if max_length is not None else tokenizer.model_max_length
        self.projection_dim = projection_dim

        self.projection = nn.Sequential(
            nn.Linear(encoder_dim, projection_dim, bias=False),
            nn.LayerNorm(projection_dim)
        ) if projection_dim is not None else None

    def forward(self, text: Sequence[str]):
        tokens = self.tokenizer(list(text), truncation=True, max_length=self.max_length, padding="max_length",
                                return_tensors="pt").input_ids  # TODO: attention_mask
        tokens = tokens.to(self.encoder.device)
        z = self.encoder(tokens).last_hidden_state
        if self.projection is not None:
            z = self.projection(z)
        return z


class CustomEmbedding:
    logger = rank_zero_logger("custom-embed")

    def __init__(self, keyword, vectors):
        self.keyword = keyword
        self.vectors = vectors
        self.tokens = [f"emb-{keyword}-{i}" for i in range(len(vectors))]
        self.keyword_regex = re.compile(fr"(?:^|(?<=\s|,)){keyword}(?=,|\s|$)")
        self.keyword_replacement = " ".join(self.tokens)

    @classmethod
    def load(cls, path: Path):
        keyword = path.stem

        assert ' ' not in keyword, f'Embedding "{keyword}": Name cannot contain spaces.'

        state = torch.load(path, map_location='cpu')
        embs = list(state['string_to_param'].values())

        assert len(embs) == 1, f'Embedding "{keyword}": Expected one embedding per file, got {len(embs)}.'

        vectors = embs[0]
        cls.logger.info("Keyword: {}, num vectors: {}", keyword, len(vectors))

        return cls(keyword, vectors)

    def expand_keyword(self, text):
        return self.keyword_regex.sub(self.keyword_replacement, text)


class CLIPTextEncoder(TextEncoder):
    def __init__(self, name: str,
                 stop_at_layer=1,
                 projection_dim: Optional[int] = None,
                 max_length: Optional[int] = None):
        from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextConfig
        config = CLIPTextConfig.from_pretrained(name)
        config.num_hidden_layers -= stop_at_layer - 1

        with TransformersNoStupidWarnings():
            clip = CLIPTextModel.from_pretrained(name, config=config)

        tokenizer = CLIPTokenizer.from_pretrained(name)

        self.custom_embeddings = None

        super().__init__(clip, tokenizer, config.hidden_size, projection_dim=projection_dim, max_length=max_length)

    def init_custom_embeddings(self, custom_embeddings: Collection[CustomEmbedding]):
        if self.custom_embeddings is not None:
            raise Exception("Already initialized")

        tokens = [
            token
            for emb in custom_embeddings
            for token in emb.tokens
        ]
        n_added = self.tokenizer.add_tokens(tokens)

        assert n_added == len(tokens), \
            f"Unexpected number of tokens added: {n_added}, expected {len(tokens)}. " \
            "Try make the emb names are less nasty."

        custom_vectors = torch.cat([emb.vectors for emb in custom_embeddings], dim=0)

        emb_layer = self.encoder.get_input_embeddings()
        vectors = torch.cat([emb_layer.weight, custom_vectors], dim=0)
        emb_layer.weight = torch.nn.Parameter(vectors, requires_grad=False)
        self.encoder.set_input_embeddings(emb_layer)

        original_prepare_for_tokenization = self.tokenizer.prepare_for_tokenization

        def prepare_for_tokenization(_tokenizer, text: str, is_split_into_words: bool = False, **kwargs):
            for emb in custom_embeddings:
                text = emb.expand_keyword(text)

            text = original_prepare_for_tokenization(text, is_split_into_words, **kwargs)
            return text

        self.tokenizer.prepare_for_tokenization = types.MethodType(prepare_for_tokenization, self.tokenizer)
        self.custom_embeddings = custom_embeddings


class EnsembleTextEncoder(nn.Module):
    def __init__(self, encoders: Sequence[TextEncoder]):
        super().__init__()
        self.encoders = nn.ModuleList(encoders)

    def forward(self, text: Sequence[str]):
        zs = [encoder(text) for encoder in self.encoders]
        z = rearrange(zs, "x b n d -> b (x n) d")
        return z


def clip_t5_encoder(clip_model=CLIP_H, clip_length=77, t5_model=T5_L, t5_length=113, projection_dim=1024):
    clip_encoder = CLIPTextEncoder(clip_model,
                                   max_length=clip_length, stop_at_layer=2, projection_dim=projection_dim)

    from transformers import T5EncoderModel, T5TokenizerFast
    t5 = T5EncoderModel.from_pretrained(t5_model)
    t5_tokenizer = T5TokenizerFast.from_pretrained(clip_model)
    t5_encoder = TextEncoder(t5, t5_tokenizer,
                             max_length=t5_length, encoder_dim=t5.config.d_model, projection_dim=projection_dim)

    return EnsembleTextEncoder([clip_encoder, t5_encoder])
