from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from omegaconf import ListConfig
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPTokenizer

from . import Size
from .augment import AugmentTransforms
from ..utils import read_image, list_images


@dataclass
class Concept:
    path: Path
    prompt: str


@dataclass
class Item:
    path: Path
    token_ids: list[int]
    image: Optional[torch.Tensor] = None
    # Cache
    latent: Optional[torch.Tensor] = None
    condition: Optional[torch.Tensor] = None


@dataclass
class Index:
    value: int
    size: Size


class ImagePromptDataset(Dataset[Item]):
    _augment_transforms: Optional[AugmentTransforms] = None

    # cached_conds = False
    # cached_latents = False

    PLACEHOLDER_TXT_PROMPT = "{TXT_PROMPT}"

    def __init__(self,
                 concepts: Collection[Concept],
                 tokenizer: CLIPTokenizer,
                 center_crop=False,
                 pad_tokens=False,
                 augment_config: Optional[ListConfig] = None):
        self.dir_prompt_map = {Path(concept.path): concept.prompt for concept in concepts}
        self.image_paths = list(list_images(*self.dir_prompt_map.keys()))

        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.pad_tokens = pad_tokens

        if augment_config is not None:
            from .augment import AugmentTransforms
            self._augment_transforms = AugmentTransforms(augment_config)

    def __getitem__(self, index: Index):
        path = self.image_paths[index.value]
        return Item(
            path=path,
            image=self._read_and_transform(path, index.size),
            token_ids=self._tokenize(self._get_prompt(path))
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def _get_prompt(self, path: Path) -> str:
        prompt = self.dir_prompt_map[path.parent]

        if prompt is None:
            prompt = self.PLACEHOLDER_TXT_PROMPT
        elif self.PLACEHOLDER_TXT_PROMPT not in prompt:
            return prompt

        txt_path = path.with_suffix('.txt')
        assert txt_path.is_file(), f'Image "{path}" does not have corresponding prompt txt'

        txt_prompt = txt_path.read_text()
        prompt = prompt.replace(self.PLACEHOLDER_TXT_PROMPT, txt_prompt)
        return prompt

    def _tokenize(self, prompt: str) -> list[int]:
        return self.tokenizer(
            prompt,
            padding="max_length" if self.pad_tokens else "do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

    def _augment(self, image: torch.Tensor) -> torch.Tensor:
        w, h = image.shape[-1], image.shape[-2]
        image = self._augment_transforms(image)
        image = transforms.Resize((h, w), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)(image)
        return image

    def _read_and_transform(self, path: Path, size: Size) -> torch.Tensor:
        # if not self.cached_latents:
        pil_image: Image.Image = read_image(path)
        dim = size[0]
        image: torch.Tensor = transforms.Compose(
            [
                transforms.Resize(dim, interpolation=transforms.InterpolationMode.LANCZOS),
                transforms.CenterCrop(dim) if self.center_crop else transforms.RandomCrop(dim),
                transforms.ToTensor()
            ]
        )(pil_image)
        if self._augment_transforms is not None:
            image = self._augment(image)
        image = transforms.Normalize([0.5], [0.5])(image)
        return image

    # def do_cache(self, vae: AutoencoderKL, text_encoder: CLIPTextModel = None):
    #     train_dataloader = torch.utils.data.DataLoader(
    #         self, shuffle=True, collate_fn=lambda x: x, pin_memory=True
    #     )
    #
    #     with torch.inference_mode():
    #         for batch in tqdm(train_dataloader):
    #             for entry in batch:
    #                 entry.latent = vae.encode(entry.image).latent_dist.sample() * 0.18215
    #                 if text_encoder is not None:
    #                     entry.cond = text_encoder.forward(entry.token_ids)
    #
    #     self.cached_latents = True
    #     if text_encoder is not None:
    #         self.cached_conds = True


class AspectDataset(ImagePromptDataset):

    def __init__(self, *args, debug=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug = debug

    def _get_transform(self, size: Size, dsize: Size):
        w_t, h_t = self._perserve_ratio_size(size, dsize)
        w_d, h_d = dsize

        return transforms.Compose([
            transforms.Resize((h_t, w_t), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop((h_d, w_d)) if self.center_crop else transforms.RandomCrop((h_d, w_d)),
            transforms.ToTensor(),
        ])

    def _read_and_transform(self, path: Path, size: Size):
        pil_image: Image.Image = read_image(path)
        image: torch.Tensor = self._get_transform(pil_image.size, size)(pil_image)

        if self.debug:
            result_wh = (image.shape[-1], image.shape[-2])
            print(f"{pil_image.size} {size} -> {result_wh}")
            import uuid, torchvision, tempfile

            save_dir = Path(tempfile.gettempdir(), "arb_debug")
            save_dir.mkdir(exist_ok=True)

            f_id = str(uuid.uuid4())

            log_new_img: Image.Image = torchvision.transforms.ToPILImage()(image)

            pil_image.save(save_dir / f"{f_id}_orig.png")
            log_new_img.save(save_dir / f"{f_id}_tr.png")

            print(f"Saved sample: {save_dir / f_id}")

        if self._augment_transforms is not None:
            image = self._augment(image)

        image = transforms.Normalize([0.5], [0.5])(image)

        return image

    @staticmethod
    def _perserve_ratio_size(size: Size, dsize: Size):
        w, h = size
        short, long = (w, h) if w <= h else (h, w)

        w_d, h_d = dsize
        min_crop, max_crop = (w_d, h_d) if w_d <= h_d else (h_d, w_d)
        ratio_src, ratio_dst = long / short, max_crop / min_crop

        if ratio_src > ratio_dst:
            w_t, h_t = (min_crop, int(min_crop * ratio_src)) if w < h else (int(min_crop * ratio_src), min_crop)
        elif ratio_src < ratio_dst:
            w_t, h_t = (max_crop, int(max_crop / ratio_src)) if w > h else (int(max_crop / ratio_src), max_crop)
        else:
            w_t, h_t = w_d, h_d

        return w_t, h_t


class DBDataset(Dataset[tuple[Item, Item]]):

    def __init__(self,
                 instance_set: ImagePromptDataset,
                 class_set: ImagePromptDataset):
        self.instance_set = instance_set
        self.class_set = class_set

    def __len__(self) -> int:
        return len(self.instance_set)

    def __getitem__(self, index: tuple[Index, Index]) -> tuple[Item, Item]:
        instance = self.instance_set[index[0]]
        class_ = self.class_set[index[1]]
        return instance, class_

    # def do_cache(self, vae: AutoencoderKL, text_encoder: CLIPTextModel = None):
    #     train_dataloader = torch.utils.data.DataLoader(
    #         self, shuffle=True, collate_fn=lambda x: x, pin_memory=True
    #     )
    #
    #     with torch.inference_mode():
    #         for batch in tqdm(train_dataloader):
    #             for entries in batch:
    #                 for entry in entries:
    #                     entry.latent = vae.encode(entry.image).latent_dist.sample() * 0.18215
    #                     if text_encoder is not None:
    #                         entry.cond = text_encoder.forward(entry.token_ids)
    #
    #     self.cached_latents = True
    #     if text_encoder is not None:
    #         self.cached_conds = True
