import sys
from pathlib import Path

import click

parent = str(Path(__file__).parent.parent.absolute())
sys.path.append(parent)

from modules.dataset import AspectDataset
from modules.dataset.bucket import BucketManager


@click.command()
@click.argument("width", type=int)
@click.argument("height", type=int)
@click.option("--dim", type=int, default=512)
def meta_single_image(width, height, dim):
    aspect = width / height
    manager = BucketManager(1, 114514)
    manager.gen_buckets(base_res=(dim, dim),
                        max_size=int(dim * dim * 1.5),
                        dim_range=(dim // 2, dim * 2),
                        divisor=int(dim / 8))
    manager.put_in({"69": (width, height)})

    best_fit = next(b for b in manager.buckets if any(b.ids))
    error = abs(best_fit.aspect - aspect)
    before_crop = AspectDataset._perserve_ratio_size((width, height), best_fit.size)

    resolutions = [bucket.size for bucket in manager.buckets]

    print(f"Buckets:\n{resolutions}")
    print(f"Best fit bucket={best_fit.size}, error={error}")
    print(f"Resize {(width, height)} -> {before_crop} before crop")


if __name__ == '__main__':
    meta_single_image()
