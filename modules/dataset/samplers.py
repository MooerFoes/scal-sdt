import copy
import random

from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Sampler

from . import Size
from .bucket import BucketManager
from .datasets import ImagePromptDataset, DBDataset, Index, AspectDataset


def scale_bucket_params(dim: int, c_size: float, c_dim: float, c_div: float):
    return {
        "base_res": (dim, dim),
        "max_size": int(dim ** 2 * c_size),
        "dim_range": (int(dim / c_dim), int(dim * c_dim)),
        "divisor": int(dim / c_div)
    }


def get_gen_bucket_params(dim: int, bucket_config: DictConfig):
    params = scale_bucket_params(
        dim,
        bucket_config.c_size,
        bucket_config.c_dim,
        bucket_config.c_div
    )

    manual = bucket_config.get("manual")

    if manual is not None:
        params = OmegaConf.merge(params, manual)

    return params


class ConstantSizeSampler(Sampler):

    def __init__(self,
                 data_source: ImagePromptDataset,
                 size: int):
        super().__init__(data_source)
        self._len = len(data_source)
        self.size = size

    def __iter__(self):
        for index in range(self._len):
            yield Index(index, (self.size, self.size))

    def __len__(self):
        return self._len


class ConstantSizeSamplerDB(Sampler):

    def __init__(self,
                 data_source: DBDataset,
                 size: int):
        super().__init__(data_source)
        self._len = len(data_source.instance_set)
        self._class_len = len(data_source.class_set)
        self.size = size

    def __iter__(self):
        size = (self.size, self.size)
        for index in range(self._len):
            yield Index(index, size), Index(random.randint(0, self._class_len - 1), size)

    def __len__(self):
        return self._len


class AspectSampler(Sampler):

    def __init__(self,
                 data_source: AspectDataset,
                 base_size: int,
                 bucket_config: DictConfig,
                 batch_size: int,
                 seed: int,
                 world_size=1,
                 global_rank=0):
        super().__init__(data_source)

        bucket_manager = BucketManager[int](batch_size, seed, world_size, global_rank)

        bucket_params = get_gen_bucket_params(base_size, bucket_config)
        bucket_manager.gen_buckets(**bucket_params)

        bucket_manager.put_in(data_source.id_size_map, bucket_config.max_aspect_error)

        self.bucket_manager = bucket_manager
        self._world_size = world_size
        self._batch_size = batch_size

    def __iter__(self):
        for batch, size in self.bucket_manager.generator():
            yield from (Index(index, size) for index in batch)

    def __len__(self):
        if self.bucket_manager.epoch_null:
            self.bucket_manager.start_epoch()

        return self.bucket_manager.batch_total * self._batch_size


class AspectSamplerDB(Sampler):

    def __init__(self,
                 data_source: DBDataset,
                 base_size: int,
                 bucket_config: DictConfig,
                 batch_size: int,
                 seed: int,
                 world_size=1,
                 global_rank=0):
        super().__init__(data_source)
        bucket_manager = BucketManager[int](batch_size, seed, world_size, global_rank)

        bucket_params = get_gen_bucket_params(base_size, bucket_config)
        bucket_manager.gen_buckets(**bucket_params)
        instance_buckets = copy.deepcopy(bucket_manager.buckets)

        bucket_manager.put_in(data_source.instance_set.id_size_map, bucket_config.max_aspect_error)

        self.bucket_manager = bucket_manager
        self._image_paths = data_source.instance_set.image_paths
        self._world_size = world_size
        self._batch_size = batch_size

        class_bucket_manager = BucketManager[int](1, seed, world_size, global_rank, False)
        class_bucket_manager.buckets = instance_buckets
        class_bucket_manager.base_res = bucket_manager.base_res

        self.class_bucket_id_map = dict[Size, list[int]]()

        class_bucket_manager.put_in(data_source.class_set.id_size_map, bucket_config.max_aspect_error)

        for batch, size in class_bucket_manager.generator():
            class_id = batch[0]
            self.class_bucket_id_map.setdefault(size, []).append(class_id)

    def __iter__(self):
        for batch, instance_size in self.bucket_manager.generator():
            for instance_id in batch:
                if not (instance_size in self.class_bucket_id_map and any(
                        self.class_bucket_id_map[instance_size])):
                    class_ids = self._get_closest_class_entries_to_size(instance_size)
                else:
                    class_ids = self.class_bucket_id_map[instance_size]

                class_id = random.choice(class_ids)

                yield Index(instance_id, instance_size), Index(class_id, instance_size)

    def __len__(self):
        if self.bucket_manager.epoch_empty:
            self.bucket_manager.start_epoch()

        return self.bucket_manager.batch_total * self._batch_size

    def _get_closest_class_entries_to_size(self, size):
        target_aspect = size[0] / size[1]

        def error(class_sz):
            aspect = class_sz[0] / class_sz[1]
            return abs(aspect - target_aspect)

        closest_size = min(self.class_bucket_id_map.keys(), key=error)
        return self.class_bucket_id_map[closest_size]
