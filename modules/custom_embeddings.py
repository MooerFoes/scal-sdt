import re
import types
from pathlib import Path

import torch
from transformers import CLIPTokenizer, CLIPTextModel


class CustomEmbeddingsHook:
    def __init__(self, path: Path | str):
        embs = {}
        for p in Path(path).glob('*.pt'):
            name, vec = self.load_emb(p)
            embs[name] = vec
        self.embs = embs
        self.clip_keywords = [' '.join(s) for s in self.make_token_names(embs)]
        self.reg_match = [re.compile(fr"(?:^|(?<=\s|,)){k}(?=,|\s|$)") for k in self.embs.keys()]

    def parse_prompt(self, prompt: str):
        """Parse a prompt string into a list of embedding names and a list of tokens.
        """
        for m, v in zip(self.reg_match, self.clip_keywords):
            prompt = m.sub(v, prompt)
        return prompt

    @staticmethod
    def load_emb(path: Path | str) -> tuple[str, torch.Tensor]:
        emb = torch.load(path, map_location='cpu')
        name = Path(path).stem
        vecs = list(emb['string_to_param'].values())

        assert len(vecs) == 1, f'Embedding "{name}": Expected 1 vector per embedding, got {len(vecs)}.'
        assert ' ' not in name, f'Embedding "{name}": Name cannot contain spaces.'

        vec = vecs[0]
        return name, vec

    @staticmethod
    def make_token_names(embs: dict[str, torch.Tensor]):
        all_tokens = []
        for name, vec in embs.items():
            tokens = [f'emb-{name}-{i}' for i in range(len(vec))]
            all_tokens.append(tokens)
        return all_tokens

    def hook_clip(self, clip: CLIPTextModel, tokenizer: CLIPTokenizer):
        """Adds custom embeddings to a CLIPTextModel. CLIPTokenizer is hooked to replace the custom embedding tokens with their corresponding CLIP tokens."""
        token_names = self.make_token_names(self.embs)
        token_names = [t for sublist in token_names for t in sublist]  # flatten nested list

        # add emb tokens to tokenizer
        n_added = tokenizer.add_tokens(token_names)
        delta_embeddings = torch.cat(list(self.embs.values()), dim=0)

        assert n_added == len(
            delta_embeddings), f"Unexpected number of tokens added: {n_added} vs {len(delta_embeddings)}. Try make the emb names are less nasty."

        # append TI embeddings to CLIP embedding table
        emb_layer = clip.get_input_embeddings()
        old_embeddings = emb_layer.weight
        new_embeddings = torch.cat([old_embeddings, delta_embeddings], dim=0)  # type: ignore
        emb_layer.weight = torch.nn.Parameter(new_embeddings)
        clip.set_input_embeddings(emb_layer)

        # hook tokenizer to replace emb tokens with their corresponding CLIP tokens
        original_prepare_for_tokenization = tokenizer.prepare_for_tokenization

        def prepare_for_tokenization(_, text: str, is_split_into_words: bool = False, **kwargs):
            text = self.parse_prompt(text)
            r = original_prepare_for_tokenization(text, is_split_into_words, **kwargs)
            return r

        tokenizer.prepare_for_tokenization = types.MethodType(prepare_for_tokenization, tokenizer)
