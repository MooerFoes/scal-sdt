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

    resolutions = sorted(buckets, key=lambda sz: sz[0] * 4096 - sz[1])

    print(f"Buckets:\n{resolutions}")


def arb_transform(source_size: tuple[int, int], size: tuple[int, int]):
    print(f"Class: {source_size}, Bucket: {size}")

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

    print(f"ARB will fuck your class image to: {(new_w, new_h)} before random crop")


gen_buckets(base_res=(512, 512), max_size=512 * 768, dim_range=(256, 1024), divisor=64)
arb_transform((512, 512), (512, 768))
