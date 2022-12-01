import sys
from pathlib import Path

import click

parent = str(Path(__file__).parent.parent.absolute())
sys.path.append(parent)

from modules.dataset.bucket import BucketManager


def gen_buckets(base_res=(512, 512), max_size=512 * 768, dim_range=(256, 1024), divisor=64):
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

    return sorted(buckets, key=lambda sz: sz[0] * 4096 - sz[1])


def arb_transform(source_size: tuple[int, int], size: tuple[int, int]):
    x, y = source_size
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

    return new_w, new_h


@click.command()
@click.argument("width", type=int)
@click.argument("height", type=int)
def main(width, height):
    aspect = width / height
    manager = BucketManager(1, 114514)
    manager.gen_buckets(base_res=(512, 512), max_size=512 * 768, dim_range=(256, 1024), divisor=64)

    best_fit = min(manager.buckets, key=lambda b: abs(b.aspect - aspect))
    error = abs(best_fit.aspect - aspect)
    before_crop = arb_transform((width, height), best_fit.size)

    resolutions = [bucket.size for bucket in manager.buckets]

    print(f"Buckets:\n{resolutions}")
    print(f"Best fit bucket={best_fit.size}, error={error}")
    print(f"Resize {(width, height)} -> {before_crop} before crop")

if __name__ == '__main__':
    main()