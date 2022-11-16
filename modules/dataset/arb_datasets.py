import random
from collections.abc import Iterable
from logging import getLogger

import torch
import torch.utils
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

from .bucket import BucketManager
from .datasets import SDDataset, DBDataset, Item

logger = getLogger("ARB")


class SDDatasetWithARB(torch.utils.data.IterableDataset, SDDataset):
    def __init__(self, batch_size=1, seed=69, debug=False, **kwargs):
        super().__init__(**kwargs)
        self.debug = debug
        self.batch_size = batch_size

        id_size_map = self.get_id_size_map(self.entries)
        self.id_entry_map: dict[str, Item] = {str(entry.path): entry for entry in self.entries}

        bucket_manager = BucketManager(batch_size=batch_size, seed=seed, debug=debug)
        bucket_manager.gen_buckets()
        bucket_manager.put_in(id_size_map)

        self.bucket_manager = bucket_manager

    def __len__(self):
        return self._length // self.batch_size

    @staticmethod
    def get_id_size_map(entries: Iterable[Item]):
        path_size_map = {}

        for entry in tqdm(entries, desc="Loading resolution from entries"):
            entry: Item
            with Image.open(entry.path) as img:
                size = img.size
            path_size_map[str(entry.path)] = size

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
            print(x, y, w, h, "->", new_img.shape)
            import uuid, torchvision
            filename = str(uuid.uuid4())
            torchvision.utils.save_image(torchvision.transforms.ToTensor()(img), f"/tmp/{filename}_orig.png")
            torchvision.utils.save_image(self.denormalize(new_img), f"/tmp/{filename}_gen.png")
            print(f"Saved sample: /tmp/{filename}")

        return new_img

    def __iter__(self):
        for batch, size in self.bucket_manager.generator():
            for id in batch:
                entry = self.id_entry_map[id]
                image = self.read_img(entry.path)
                entry.image = self.transform(image, size)
                yield entry


class DBDatasetWithARB(DBDataset, SDDatasetWithARB):
    def __init__(self, batch_size=1, seed=69, debug=False, **kwargs):
        super().__init__(batch_size, seed, debug, **kwargs)

        self.class_bucket_entry_map = dict[tuple[int, int], list[Item]]()

        class_id_size_map = self.get_id_size_map(self.class_entries)
        class_id_entry_map: dict[str, Item] = {str(entry.path): entry for entry in self.class_entries}

        class_bucket_manager = BucketManager(batch_size=1, seed=seed, debug=debug)
        class_bucket_manager.buckets = self.bucket_manager.buckets
        class_bucket_manager.base_res = self.bucket_manager.base_res

        class_bucket_manager.put_in(class_id_size_map)

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
        for batch, instance_size in super().bucket_manager.generator():
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
