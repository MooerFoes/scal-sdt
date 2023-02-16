import json
import random
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Optional, Any

import torch
from PIL import Image
from omegaconf import ListConfig
from safetensors import safe_open
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from . import Size
from .augment import AugmentTransforms
from ..utils.io.image import read_image, list_images


@dataclass
class Concept:
    path: Path
    prompt: str


@dataclass
class TextConditionalItem:
    id: int
    text: str
    image: torch.Tensor


@dataclass
class TextConditionalItemCached:
    id: int
    latent: torch.Tensor
    condition: torch.Tensor


ItemTypeTextConditional = TextConditionalItem | TextConditionalItemCached


@dataclass
class TextImageConditionalItem(TextConditionalItem):
    cond_image: torch.Tensor


@dataclass
class PriorPreservationItem:
    instance: ItemTypeTextConditional
    prior: ItemTypeTextConditional


ItemTypeAll = ItemTypeTextConditional | PriorPreservationItem | TextImageConditionalItem


@dataclass
class Index:
    value: int
    size: Size


class TextConditionalDataset(Dataset[ItemTypeTextConditional]):
    PLACEHOLDER_TXT_TEXT = "{TXT_PROMPT}"

    def __init__(self,
                 concepts: Collection[Concept],
                 center_crop=False,
                 augment_config: Optional[ListConfig] = None,
                 cache_file: Optional[str | PathLike] = None):
        self.dir_prompt_map = {Path(concept.path): concept.prompt for concept in concepts}
        self.image_paths = self._list_images(self.dir_prompt_map)
        self.center_crop = center_crop

        self.cache: Optional[safe_open] = None
        self._cache_metadata: Optional[dict[str, Any]] = None
        self._augment_transforms: Optional[AugmentTransforms] = None

        if augment_config is not None:
            self._augment_transforms = AugmentTransforms(augment_config)

        if cache_file is not None:
            self.cache = safe_open(cache_file, framework="pt", device="cpu")
            self._cache_metadata = json.loads(self.cache.metadata()["json"])

    def _list_images(self, dir_prompt_map):
        return list(list_images(*dir_prompt_map.keys()))

    def __getitem__(self, index: Index):
        if self.cache is None:
            path = self.image_paths[index.value]
            return TextConditionalItem(
                id=index.value,
                image=self._augment(self._read_and_transform(path, index.size)),
                text=self._get_text(path)
            )
        else:
            return TextConditionalItemCached(
                id=index.value,
                latent=self.cache.get_tensor(
                    f"{index.value}.latent.{random.randint(0, self._cache_metadata['aug_group_size'] - 1)}"),
                condition=self.cache.get_tensor(f"{index.value}.cond")
            )

    def __len__(self) -> int:
        return len(self.image_paths) if self.cache is None else self._cache_metadata["total_entries"]

    def _get_text(self, path: Path) -> str:
        text = self.dir_prompt_map[path.parent]

        if text is None:
            text = self.PLACEHOLDER_TXT_TEXT
        elif self.PLACEHOLDER_TXT_TEXT not in text:
            return text

        txt_path = path.with_suffix('.txt')
        assert txt_path.is_file(), f'Image "{path}" does not have corresponding prompt txt'

        txt_prompt = txt_path.read_text()
        text = text.replace(self.PLACEHOLDER_TXT_TEXT, txt_prompt)
        return text

    def _augment(self, image: torch.Tensor) -> torch.Tensor:
        if self._augment_transforms is None:
            return image

        w, h = image.shape[-1], image.shape[-2]
        image = self._augment_transforms(image)
        image = transforms.Resize((h, w), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)(image)
        return image

    def _read_and_transform(self, path: Path, size: Size) -> torch.Tensor:
        pil_image: Image.Image = read_image(path)
        dim = size[0]
        image: torch.Tensor = transforms.Compose(
            [
                transforms.Resize(dim, interpolation=transforms.InterpolationMode.LANCZOS),
                transforms.CenterCrop(dim) if self.center_crop else transforms.RandomCrop(dim),
                transforms.ToTensor()
            ]
        )(pil_image)
        image = transforms.Normalize([0.5], [0.5])(image)
        return image


class TextImageConditionalDataset(TextConditionalDataset):
    SUFFIX_COND_IMAGE = "_cond-image"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _list_images(self, dir_prompt_map):
        return list(filter(lambda x: not x.stem.endswith(self.SUFFIX_COND_IMAGE),
                           list_images(*self.dir_prompt_map.keys())))

    def __getitem__(self, index: Index):
        if self.cache is not None:
            raise NotImplementedError("Cache is not implemented")

        path = self.image_paths[index.value]
        return TextImageConditionalItem(
            id=index.value,
            image=self._augment(self._read_and_transform(path, index.size)),
            text=self._get_text(path),
            cond_image=self._get_cond_image(path, index.size)
        )

    def _get_cond_image(self, path: Path, size: Size) -> torch.Tensor:
        cond_image = path.with_stem(path.stem + self.SUFFIX_COND_IMAGE)
        return self._read_and_transform(cond_image, size)


def get_id_size_map(image_paths: Iterable[Path]):
    id_size_map = {}
    for i, path in enumerate(tqdm(image_paths, desc="Loading resolution from entries")):
        path: Path
        with Image.open(path) as img:
            size = img.size
        id_size_map[i] = size
    return id_size_map


class AspectTextConditionalDataset(TextConditionalDataset):
    def __init__(self, *args, debug=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug = debug

        self.id_size_map: dict[int, Size]()

        if self.cache is None:
            self.id_size_map = get_id_size_map(self.image_paths)
        else:
            sizes_info = self._cache_metadata["sizes"]
            for k in self._cache_metadata["entries"]:
                self.id_size_map[k] = Size(sizes_info[f"{k}.latent.0"])

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


class AspectTextImageConditionalDataset(TextImageConditionalDataset, AspectTextConditionalDataset):
    pass


class PriorPreservationDataset(Dataset[PriorPreservationItem]):

    def __init__(self,
                 instance_set: TextConditionalDataset | AspectTextConditionalDataset,
                 class_set: TextConditionalDataset | AspectTextConditionalDataset):
        self.instance_set = instance_set
        self.class_set = class_set

    def __len__(self) -> int:
        return len(self.instance_set)

    def __getitem__(self, index: tuple[Index, Index]) -> PriorPreservationItem:
        instance = self.instance_set[index[0]]
        prior = self.class_set[index[1]]
        return PriorPreservationItem(instance, prior)
