import copy
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from omegaconf import ListConfig
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPTokenizer

from modules.dataset.augment import AugmentTransforms


@dataclass
class Item:
    path: Path
    token_ids: list[int]
    # Cache
    image: torch.Tensor | None = None
    latent: torch.Tensor | None = None
    condition: torch.Tensor | None = None


class SDDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """
    augment_transforms: AugmentTransforms | None = None

    # cached_conds = False
    # cached_latents = False

    def __init__(
            self,
            concepts,
            tokenizer: CLIPTokenizer,
            size=512,
            center_crop=False,
            pad_tokens=False,
            augment_config: ListConfig | None = None,
            **kwargs
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.pad_tokens = pad_tokens

        self.entries = list[Item]()

        for concept in concepts:
            instance_entries = self.resolve_dataset(concept.instance_set)
            self.entries.extend(instance_entries)

        random.shuffle(self.entries)

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.LANCZOS),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor()
            ]
        )

        if augment_config is not None:
            from .augment import AugmentTransforms
            self.augment_transforms = AugmentTransforms(augment_config)

    def tokenize(self, prompt: str) -> list[int]:
        return self.tokenizer(
            prompt,
            padding="max_length" if self.pad_tokens else "do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

    @staticmethod
    def get_images(path: Path):
        assert path.is_dir()
        return (x for x in path.iterdir() if x.is_file() and x.suffix != ".txt")

    def resolve_dataset(self, dataset):
        for x in self.get_images(Path(dataset.path)):
            prompt = dataset.prompt

            if "{TXT_PROMPT}" in prompt:
                txt_prompt = x.with_suffix('.txt').read_text()
                prompt = prompt.replace("{TXT_PROMPT}", txt_prompt)

            token_ids = self.tokenize(prompt)

            yield Item(path=x, token_ids=token_ids)

    # def do_cache(self, vae: AutoencoderKL, text_encoder: CLIPWithSkip = None):
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

    def __len__(self):
        return len(self.entries)

    def augment(self, image: torch.Tensor):
        w, h = image.shape[-1], image.shape[-2]
        image = self.augment_transforms(image)
        image = transforms.Resize((h, w), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)(image)
        return image

    def read_and_transform(self, path):
        # if not self.cached_latents:
        image = self.read_img(path)
        image = self.image_transforms(image)
        if self.augment_transforms is not None:
            image = self.augment(image)
        image = transforms.Normalize([0.5], [0.5])(image)
        return image

    def _get_item(self, entry):
        entry = copy.copy(entry)
        entry.image = self.read_and_transform(entry.path)
        return entry

    def __getitem__(self, index) -> Item:
        return self._get_item(self.entries[index])

    @staticmethod
    def read_img(filepath: Path) -> Image.Image:
        img = Image.open(filepath)

        if not img.mode == "RGB":
            img = img.convert("RGB")
        return img


class DBDataset(SDDataset):

    def __init__(self,
                 concepts,
                 tokenizer: CLIPTokenizer,
                 size=512,
                 center_crop=False,
                 pad_tokens=False,
                 **kwargs):
        super().__init__(concepts, tokenizer, size, center_crop, pad_tokens, **kwargs)

        self.class_entries = list[Item]()

        for concept in concepts:
            class_entries = self.resolve_dataset(concept.class_set)
            self.class_entries.extend(class_entries)

    def __getitem__(self, index) -> tuple[Item, Item]:
        instance = super().__getitem__(index)
        class_ = super()._get_item(random.choice(self.class_entries))
        return instance, class_

    # def do_cache(self, vae: AutoencoderKL, text_encoder: CLIPWithSkip = None):
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


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example
