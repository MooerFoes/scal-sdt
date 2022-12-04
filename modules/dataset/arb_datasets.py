import random
from collections.abc import Iterable
from pathlib import Path

import torch
import torch.utils
from PIL import Image
from omegaconf import DictConfig
from torchvision import transforms
from tqdm.auto import tqdm

from .bucket import BucketManager
from .datasets import SDDataset, DBDataset, Item


def scale_bucket_params(dim: int, c_size: float, c_dim: float, c_div: float):
    return {
        "base_res": (dim, dim),
        "max_size": int(dim ** 2 * c_size),
        "dim_range": (int(dim / c_dim), int(dim * c_dim)),
        "divisor": int(dim / c_div)
    }


def get_gen_bucket_params(dim: int, bucket_config: DictConfig):
    bucket_params = bucket_config.get("manual")
    if bucket_params is None:
        bucket_params = scale_bucket_params(
            dim,
            bucket_config.c_size,
            bucket_config.c_dim,
            bucket_config.c_div
        )
    else:
        for k, v in bucket_params:
            if isinstance(v, tuple):
                bucket_params[k] = [x for x in v]
    return bucket_params


class SDDatasetWithARB(torch.utils.data.IterableDataset, SDDataset):
    def __init__(self, bucket_config: DictConfig, batch_size=1, seed=69, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size

        id_size_map = self.get_id_size_map((entry.path for entry in self.entries))
        self.id_entry_map: dict[str, Item] = {str(entry.path): entry for entry in self.entries}

        bucket_manager = BucketManager(batch_size=batch_size, seed=seed, debug=bucket_config.debug)

        bucket_params = get_gen_bucket_params(self.size, bucket_config)
        bucket_manager.gen_buckets(**bucket_params)
        bucket_manager.put_in(id_size_map, bucket_config.max_aspect_error)

        self.bucket_manager = bucket_manager

    # def __len__(self):
    #     return self._length // self.batch_size

    @staticmethod
    def get_id_size_map(paths: Iterable[Path]):
        path_size_map = {}

        for path in tqdm(paths, desc="Loading resolution from entries"):
            path: Path
            with Image.open(path) as img:
                size = img.size
            path_size_map[str(path)] = size

        return path_size_map

    @staticmethod
    def denormalize(img, mean=0.5, std=0.5):
        res = transforms.Normalize(-1 * mean / std, 1.0 / std)(img.squeeze(0))
        res = torch.clamp(res, 0, 1)
        return res

    def transform(self, img: Image, size: tuple[int, int]):
        x, y = img.size
        short, long = (x, y) if x <= y else (y, x)

        w, h = size
        min_crop, max_crop = (w, h) if w <= h else (h, w)
        ratio_src, ratio_dst = long / short, max_crop / min_crop

        if ratio_src > ratio_dst:
            new_w, new_h = (min_crop, int(min_crop * ratio_src)) if x < y else (int(min_crop * ratio_src), min_crop)
        elif ratio_src < ratio_dst:
            new_w, new_h = (max_crop, int(max_crop / ratio_src)) if x > y else (int(max_crop / ratio_src), max_crop)
        else:
            new_w, new_h = w, h

        image_transforms = transforms.Compose([
            transforms.Resize((new_h, new_w), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop((h, w)) if self.center_crop else transforms.RandomCrop((h, w)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        new_img = image_transforms(img)

        if self.debug:
            print(f"{(x, y)} {(w, h)} -> {new_img.shape}")
            import uuid, torchvision, tempfile

            save_dir = Path(tempfile.gettempdir(), "arb_debug")
            save_dir.mkdir(exist_ok=True)

            f_id = str(uuid.uuid4())

            log_new_img: Image.Image = torchvision.transforms.ToPILImage()(self.denormalize(new_img))

            img.save(save_dir / f"{f_id}_orig.png")
            log_new_img.save(save_dir / f"{f_id}_tr.png")

            print(f"Saved sample: {save_dir / f_id}")

        if self.augment_transforms is not None:
            new_img = self.augment_transforms(new_img)

        return new_img

    def __iter__(self):
        for batch, size in self.bucket_manager.generator():
            for id in batch:
                entry = self.id_entry_map[id]
                image = self.read_img(entry.path)
                entry.image = self.transform(image, size)
                yield entry


class DBDatasetWithARB(SDDatasetWithARB, DBDataset):
    def __init__(self, bucket_config, batch_size=1, seed=69, **kwargs):
        super().__init__(bucket_config, batch_size, seed, **kwargs)

        self.class_bucket_entry_map = dict[tuple[int, int], list[Item]]()

        class_id_size_map = self.get_id_size_map((entry.path for entry in self.class_entries))
        class_id_entry_map: dict[str, Item] = {str(entry.path): entry for entry in self.class_entries}

        class_bucket_manager = BucketManager(batch_size=1, seed=seed, debug=bucket_config.debug)
        class_bucket_manager.buckets = self.bucket_manager.buckets
        class_bucket_manager.base_res = self.bucket_manager.base_res

        class_bucket_manager.put_in(class_id_size_map, max_aspect_error=bucket_config.max_aspect_error)

        for batch, size in class_bucket_manager.generator():
            class_id = batch[0]
            class_entry = class_id_entry_map[class_id]
            self.class_bucket_entry_map.setdefault(size, []).append(class_entry)

    def get_closest_class_entries_to_size(self, size):
        error_bucket_map = [(abs(class_sz[0] / class_sz[1] - size[0] / size[1]), bucket_entries)
                            for class_sz, bucket_entries in self.class_bucket_entry_map.items() if any(bucket_entries)]
        error_bucket_map.sort(key=lambda x: x[0])
        _, entries = error_bucket_map[0]
        return entries

    def __iter__(self):
        for batch, instance_size in self.bucket_manager.generator():
            for instance_id in batch:
                instance_entry = self.id_entry_map[instance_id]
                instance_image = self.read_img(instance_entry.path)
                instance_entry.image = self.transform(instance_image, instance_size)

                class_entry: Item
                if not (instance_size in self.class_bucket_entry_map and any(
                        self.class_bucket_entry_map[instance_size])):
                    entries = self.get_closest_class_entries_to_size(instance_size)
                    class_entry = random.choice(entries)
                else:
                    class_entry = random.choice(self.class_bucket_entry_map[instance_size])

                class_image = self.read_img(class_entry.path)
                class_entry.image = self.transform(class_image, instance_size)

                yield instance_entry, class_entry
