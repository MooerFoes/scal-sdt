import time
from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Optional, Generic, TypeVar

import numpy as np

from . import Size


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
    buckets: list[Bucket]
    id_size_map: dict[T_id, Size] = {}
    base_res: Size
    epoch: Optional[dict[Bucket, list[T_id]]] = None
    left_over: Optional[list[T_id]] = None
    batch_total = 0
    batch_delivered = 0

    def __init__(self, batch_size: int, seed: int, world_size=1, global_rank=0, debug=False):
        self.batch_size = batch_size
        self.world_size = world_size
        self.global_rank = global_rank
        self.prng = self.get_prng(seed)
        epoch_seed = self.prng.tomaxint() % (2 ** 32 - 1)
        self.epoch_prng = self.get_prng(epoch_seed)  # separate prng for sharding use for increased thread resilience
        self.debug = debug

    def __len__(self):
        return len(self.id_size_map) // self.batch_size

    @property
    def epoch_empty(self):
        return self.batch_delivered >= self.batch_total

    @staticmethod
    def get_prng(seed: int):
        return np.random.RandomState(seed)

    def gen_buckets(self, base_res=(512, 512), max_size=768 * 512, dim_range=(256, 1024), divisor=64):
        if self.debug:
            timer = time.perf_counter()

        self.base_res = base_res
        min_dim, max_dim = dim_range
        buckets = set()

        w = min_dim
        while w * min_dim <= max_size and w <= max_dim:
            h = min_dim
            got_base = False
            while w * (h + divisor) <= max_size and (h + divisor) <= max_dim:
                if w == base_res[0] and h == base_res[1]:
                    got_base = True
                h += divisor
            if (w != base_res[0] or h != base_res[1]) and got_base:
                buckets.add(base_res)
            buckets.add((w, h))
            w += divisor

        h = min_dim
        while h / min_dim <= max_size and h <= max_dim:
            w = min_dim
            while h * (w + divisor) <= max_size and (w + divisor) <= max_dim:
                w += divisor
            buckets.add((w, h))
            h += divisor

        resolutions = sorted(buckets, key=lambda sz: sz[0] * 4096 - sz[1])
        self.buckets = [Bucket(res) for res in resolutions]

        if self.debug:
            timer = time.perf_counter() - timer
            print(f"Bucket sizes:\n{resolutions}")
            print(f"Time: {timer:.5f}s")

    def put_in(self, id_size_map: dict[T_id, Size], max_aspect_error=0.5):
        if self.debug:
            timer = time.perf_counter()

        self.id_size_map = id_size_map
        aspect_errors = []
        skipped_ids = []

        for id, (w, h) in id_size_map.items():
            aspect = float(w) / float(h)
            best_fit = min(self.buckets, key=lambda b: abs(b.aspect - aspect))
            error = abs(best_fit.aspect - aspect)
            if error < max_aspect_error:
                best_fit.ids.append(id)
                if self.debug:
                    aspect_errors.append(error)
            else:
                skipped_ids.append(id)

        if self.debug:
            timer = time.perf_counter() - timer
            aspect_errors = np.array(aspect_errors)

            print(f"""Bucket Assignment: {timer:.5f}s
Aspect Error: mean {aspect_errors.mean()}, median {np.median(aspect_errors)}, max {aspect_errors.max()}
Skipped Images: {skipped_ids}
""")
            for bucket in self.buckets:
                print(
                    f"Bucket {bucket.size}, aspect {bucket.aspect:.5f}, {len(bucket.ids)} entries"
                )

    def start_epoch(self):
        if self.debug:
            timer = time.perf_counter()

        self.epoch = {}
        self.left_over = []
        self.batch_delivered = 0

        # select ids for this epoch/rank
        index = list(self.id_size_map.keys())
        index_len = len(index)
        index = self.epoch_prng.permutation(index)
        index = index[:index_len - (index_len % (self.batch_size * self.world_size))]
        # print("perm", self.global_rank, index[0:16])
        index = index[self.global_rank::self.world_size]
        self.batch_total = len(index) // self.batch_size
        assert (len(index) % self.batch_size == 0)
        index = set(index)

        for bucket in self.buckets:
            if not any(bucket.ids):
                continue

            self.epoch[bucket] = [id for id in bucket.ids if id in index]
            self.prng.shuffle(self.epoch[bucket])
            self.epoch[bucket] = list(self.epoch[bucket])
            overhang = len(self.epoch[bucket]) % self.batch_size
            if overhang != 0:
                self.left_over.extend(self.epoch[bucket][:overhang])
                self.epoch[bucket] = self.epoch[bucket][overhang:]
            if len(self.epoch[bucket]) == 0:
                del self.epoch[bucket]

        if self.debug:
            timer = time.perf_counter() - timer
            count = 0
            for bucket_id in self.epoch.keys():
                count += len(self.epoch[bucket_id])
            print(f"correct item count: {count == len(index)} ({count} of {len(index)})")
            print(f"start_epoch: {timer:.5f}s")

    def get_batch(self):
        if self.debug:
            timer = time.perf_counter()
        # check if no data left or no epoch initialized
        if (self.epoch is None or self.left_over is None or
                (len(self.left_over) == 0 and not any(self.epoch))
                or self.batch_total == self.batch_delivered):
            self.start_epoch()

        resolution = self.base_res
        found_batch = False
        batch_buckets = list[T_id]()
        chosen_bucket: Optional[Bucket] = None

        while not found_batch:
            buckets: list[Optional[Bucket]] = list(self.epoch.keys())
            if len(self.left_over) >= self.batch_size:
                bucket_probs = [len(self.left_over)] + [len(self.epoch[bucket]) for bucket in buckets]
                buckets = [None] + buckets
            else:
                bucket_probs = [len(self.epoch[bucket_id]) for bucket_id in buckets]
            bucket_probs = np.array(bucket_probs, dtype=np.float32)
            bucket_probs = bucket_probs / bucket_probs.sum()

            chosen_bucket = self.prng.choice(buckets, 1, p=bucket_probs)[0] if any(self.epoch) else None

            if chosen_bucket is None:
                # using leftover images that couldn't make it into a bucketed batch and returning them for use with basic square image
                self.prng.shuffle(self.left_over)
                batch_buckets = self.left_over[:self.batch_size]
                self.left_over = self.left_over[self.batch_size:]
                found_batch = True
            else:
                if len(self.epoch[chosen_bucket]) >= self.batch_size:
                    # return bucket batch and resolution
                    batch_buckets = self.epoch[chosen_bucket][:self.batch_size]
                    self.epoch[chosen_bucket] = self.epoch[chosen_bucket][self.batch_size:]
                    resolution = chosen_bucket.size
                    found_batch = True
                    if len(self.epoch[chosen_bucket]) == 0:
                        del self.epoch[chosen_bucket]
                else:
                    # can't make a batch from this, not enough images. move them to leftovers and try again
                    self.left_over.extend(self.epoch[chosen_bucket])
                    del self.epoch[chosen_bucket]

            assert (found_batch or len(self.left_over) >= self.batch_size or bool(self.epoch))

        if self.debug:
            timer = time.perf_counter() - timer
            print(f"Bucket probs: " + ", ".join(map(lambda x: f"{x:.2f}", list(bucket_probs * 100))))
            print(
                f"""Chosen bucket: {chosen_bucket}
Batch data: {batch_buckets}
get_batch: {timer:.5f}s"""
            )

        self.batch_delivered += 1
        return batch_buckets, resolution

    def generator(self):
        if self.epoch_empty:
            self.start_epoch()
        while not self.epoch_empty:
            yield self.get_batch()
