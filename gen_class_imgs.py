import hashlib
import logging
from pathlib import Path
from typing import IO

import click
import torch
from PIL import Image
from diffusers.pipelines import StableDiffusionPipeline
from omegaconf import DictConfig
from tqdm import tqdm

from modules.configs import load_with_defaults
from modules.dataset import Size
from modules.dataset.bucket import BucketManager
from modules.dataset.datasets import get_id_size_map
from modules.dataset.samplers import get_gen_bucket_params
from modules.model import LatentDiffusionModel
from modules.utils.io.image import list_images

logger = logging.getLogger("cls-gen")


def get_size_dist(image_dir: Path):
    image_paths = list(list_images(image_dir))
    image_count = len(image_paths)

    id_size_map = get_id_size_map(image_paths)

    size_dist = {}
    for size in id_size_map.values():
        if size in size_dist:
            size_dist[size] += 1
        else:
            size_dist[size] = 1

    size_dist = {k: v / image_count for k, v in size_dist.items()}

    return size_dist


def get_arb_size_dist(image_dir: Path, resolution: int, arb_config: DictConfig):
    image_paths = list(list_images(image_dir))
    image_count = len(image_paths)

    id_size_map = get_id_size_map(image_paths)

    bucket_manager = BucketManager[int](1)
    gen_bucket_params = get_gen_bucket_params(resolution, arb_config)
    bucket_manager.gen_buckets(**gen_bucket_params)

    bucket_manager.put_in(id_size_map, arb_config.max_aspect_error)

    size_dist = {bucket.size: len(bucket.ids) / image_count for bucket in
                 bucket_manager.buckets}

    return size_dist


def get_delta_dist(current_dist: dict[Size, float], target_dist: dict[Size, float]):
    dist_diff = {}

    for size, t_p in target_dist.items():
        c_p = current_dist.get(size, 0)
        if t_p > c_p:
            dist_diff[size] = t_p - c_p

    return dist_diff


def generate_class_images(pipeline: StableDiffusionPipeline, class_config: DictConfig, target_dist: dict[Size, float]):
    autogen_config = class_config.auto_generate

    image_dir = Path(class_config.path)
    image_dir.mkdir(parents=True, exist_ok=True)

    current_dist = get_size_dist(image_dir)
    dist_diff = get_delta_dist(current_dist, target_dist)
    size_count_map = {size: round(autogen_config.num_target * p) for size, p in dist_diff.items()}
    num_new_images = sum(size_count_map.values())

    logger.info(f"Current distribution:\n{current_dist}")
    logger.info(f"Target distribution:\n{target_dist}")
    logger.info(f"Distribution diff:\n{dist_diff}")
    logger.info(f"Total number of class images to sample: {num_new_images}")

    # torch.autocast("cuda") causes VAE encode fucked.
    with torch.inference_mode(), \
            tqdm(total=num_new_images, desc="Generating class images") as progress:
        for (w, h), count in size_count_map.items():
            progress.set_postfix({"size": (w, h)})

            while True:
                actual_bs = count if count - autogen_config.batch_size < 0 \
                    else autogen_config.batch_size

                if actual_bs <= 0:
                    break

                images: list[Image.Image] = pipeline(
                    prompt=class_config.prompt,
                    negative_prompt=autogen_config.negative_prompt,
                    guidance_scale=autogen_config.cfg_scale,
                    num_inference_steps=autogen_config.steps,
                    num_images_per_prompt=actual_bs,
                    width=w,
                    height=h
                ).images

                for image in images:
                    hash = hashlib.md5(image.tobytes()).hexdigest()
                    image_filename = image_dir / f"{hash}.png"
                    image.save(image_filename)

                progress.update(actual_bs)
                count -= actual_bs


@click.command()
@click.option("--config", "config_file", type=click.File("r"))
def main(config_file: IO[str]):
    config = load_with_defaults(config_file)

    if not config.prior_preservation.enabled:
        logger.warning("Prior preservation not enabled. Class image generation is not needed.")
        return

    pipeline = LatentDiffusionModel.from_config(config).pipeline
    pipeline.unet.to(torch.float16)
    pipeline.to("cuda")

    arb_config = config.aspect_ratio_bucket

    for i, concept in enumerate(config.data.concepts):
        class_config = concept.class_set

        if not class_config.auto_generate.enabled:
            logger.warning(f"Concept [{i}] skipped because class auto generate is not enabled.")
            continue

        resolution = config.data.resolution

        if arb_config.enabled:
            instance_path = Path(concept.instance_set.path)
            size_dist = get_arb_size_dist(instance_path, resolution, arb_config)
        else:
            size_dist = {(resolution, resolution): 1.0}

        generate_class_images(pipeline, class_config, size_dist)


if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    main()
