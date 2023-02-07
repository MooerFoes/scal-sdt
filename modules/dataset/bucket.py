from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Optional, Generic, TypeVar

import numpy as np

from . import Size
from ..utils.logging import rank_zero_logger

logger = rank_zero_logger("arb")


@dataclass
class Bucket:
    size: Size
    ids: list[Hashable] = field(default_factory=list[Hashable])

    def __hash__(self) -> int:
        return self.size.__hash__()

    def __str__(self) -> str:
        return str(self.size)

    @property
    def aspect(self):
        return float(self.size[0]) / float(self.size[1])


T_id = TypeVar("T_id", bound=Hashable)


class BucketManager(Generic[T_id]):

    def __init__(self, batch_size: int, seed: Optional[int] = None, world_size=1, global_rank=0):
        self.batch_size = batch_size
        self.world_size = world_size
        self.global_rank = global_rank

        self.buckets: Optional[list[Bucket]] = None
        self.id_size_map: dict[T_id, Size] = {}
        self.base_res: Optional[Size] = None
        self.epoch: Optional[dict[Bucket, list[T_id]]] = None
        self.epoch_remainders: Optional[list[T_id]] = None
        self.batch_total = 0
        self.batch_delivered = 0

        self.bucket_prng = np.random.RandomState(seed)
        # separate prng for sharding use for increased thread resilience
        sharding_seed = self.bucket_prng.tomaxint() % (2 ** 32 - 1)
        self.sharding_prng = np.random.RandomState(sharding_seed)

    @property
    def epoch_null(self):
        return self.epoch is None or self.epoch_remainders is None

    @property
    def epoch_empty(self):
        return not (any(self.epoch_remainders) or any(self.epoch)) or self.batch_total == self.batch_delivered

    def gen_buckets(self, base_res=(512, 512), max_size=768 * 512, dim_range=(256, 1024), divisor=64):
        min_dim, max_dim = dim_range
        resolutions = set()

        w = min_dim
        while w * min_dim <= max_size and w <= max_dim:
            h = min_dim
            while w * (h + divisor) <= max_size and (h + divisor) <= max_dim:
                if (w, h) == base_res:
                    resolutions.add(base_res)
                h += divisor
            resolutions.add((w, h))
            w += divisor

        h = min_dim
        while h / min_dim <= max_size and h <= max_dim:
            w = min_dim
            while h * (w + divisor) <= max_size and (w + divisor) <= max_dim:
                w += divisor
            resolutions.add((w, h))
            h += divisor

        self.base_res = base_res
        self.buckets = [Bucket(res) for res in sorted(resolutions)]

        logger.debug("Bucket sizes: {}", resolutions)

    def put_in(self, id_size_map: dict[T_id, Size], max_aspect_error=0.5):
        self.id_size_map = id_size_map
        aspect_errors = []
        skipped_ids = []

        for id, (w, h) in id_size_map.items():
            aspect = float(w) / float(h)
            best_fit = min(self.buckets, key=lambda b: abs(b.aspect - aspect))
            error = abs(best_fit.aspect - aspect)
            if error < max_aspect_error:
                best_fit.ids.append(id)
                aspect_errors.append(error)
            else:
                skipped_ids.append(id)

        aspect_errors = np.array(aspect_errors)

        logger.debug("Aspect Error: mean {}, median {}, max {}",
                     np.mean(aspect_errors), np.median(aspect_errors), np.max(aspect_errors))
        logger.debug("Skipped Entries: {}", skipped_ids)
        for bucket in self.buckets:
            logger.debug("Bucket {}, aspect {:.5f}, {} entries", bucket.size, bucket.aspect, len(bucket.ids))

    def _get_local_ids(self):
        """Select ids of an epoch for this local rank."""
        local_ids = list(self.id_size_map.keys())
        index_len = len(local_ids)
        self.sharding_prng.shuffle(local_ids)

        local_ids = local_ids[:index_len - (index_len % (self.batch_size * self.world_size))]
        local_ids = local_ids[self.global_rank::self.world_size]

        index_len = len(local_ids)
        self.batch_total = index_len // self.batch_size
        assert (index_len % self.batch_size == 0)

        local_ids = set(local_ids)
        return local_ids

    def start_epoch(self):
        local_ids = self._get_local_ids()
        epoch = {}
        epoch_remainders = []

        for bucket in self.buckets:
            if not any(bucket.ids):
                continue

            chosen_ids = [id for id in bucket.ids if id in local_ids]
            self.bucket_prng.shuffle(chosen_ids)

            remainder = len(chosen_ids) % self.batch_size
            if remainder != 0:
                chosen_ids, remainders = chosen_ids[remainder:], chosen_ids[:remainder]
                epoch_remainders.extend(remainders)

            if not any(chosen_ids):
                continue

            epoch[bucket] = chosen_ids

        logger.debug("Correct item: {} / {}", sum(len(ids) for ids in epoch.values()), len(local_ids))

        self.epoch = epoch
        self.epoch_remainders = epoch_remainders
        self.batch_delivered = 0

    def get_batch(self):
        if self.epoch_null:
            raise Exception("No epoch")

        resolution = self.base_res
        found_batch = False
        batch_buckets = list[T_id]()
        chosen_bucket: Optional[Bucket] = None

        while not found_batch:
            buckets: list[Bucket | str] = list(self.epoch.keys())

            bucket_probs = [len(self.epoch[bucket_id]) for bucket_id in buckets]

            if len(self.epoch_remainders) >= self.batch_size:
                buckets.append("left_over")
                bucket_probs.append(len(self.epoch_remainders))

            bucket_probs = np.array(bucket_probs, dtype=np.float32)
            # Buckets with more images get more weight
            bucket_probs /= bucket_probs.sum()

            chosen_bucket = self.bucket_prng.choice(buckets, 1, p=bucket_probs)[0] if any(self.epoch) else "left_over"

            if chosen_bucket == "left_over":
                # using leftover images that couldn't make it into a bucketed batch and returning them for use with basic square image
                chosen_ids = self.epoch_remainders
                self.bucket_prng.shuffle(chosen_ids)
                self.epoch_remainders, batch_buckets = \
                    chosen_ids[self.batch_size:], chosen_ids[:self.batch_size]
                found_batch = True
            else:
                chosen_ids = self.epoch[chosen_bucket]
                if len(chosen_ids) >= self.batch_size:
                    # return bucket batch and resolution
                    self.epoch[chosen_bucket], batch_buckets = \
                        chosen_ids[self.batch_size:], chosen_ids[:self.batch_size]
                    resolution = chosen_bucket.size
                    found_batch = True
                    if not any(self.epoch[chosen_bucket]):
                        del self.epoch[chosen_bucket]
                else:
                    # can't make a batch from this, not enough images. move them to leftovers and try again
                    self.epoch_remainders.extend(chosen_ids)
                    del self.epoch[chosen_bucket]

            assert (found_batch or len(self.epoch_remainders) >= self.batch_size or any(self.epoch))

        logger.debug("Bucket probs: {}", ", ".join(map(lambda x: f"{x:.2f}%", list(bucket_probs * 100))))
        logger.debug("Chosen bucket: {}", chosen_bucket)
        logger.debug("Batch data", batch_buckets)

        self.batch_delivered += 1
        return batch_buckets, resolution

    def generator(self):
        if self.epoch_null or self.epoch_empty:
            self.start_epoch()

        while not self.epoch_empty:
            yield self.get_batch()
