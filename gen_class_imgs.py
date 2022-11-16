import hashlib
import logging
from pathlib import Path

import click
import torch
from diffusers.pipelines import StableDiffusionPipeline
from omegaconf import OmegaConf
from tqdm import tqdm

from modules.dataset.arb_datasets import SDDatasetWithARB
from modules.dataset.bucket import BucketManager
from modules.dataset.datasets import SDDataset

logger = logging.getLogger("cls-gen")


def generate_class_images(pipeline: StableDiffusionPipeline, concept, size_dist: dict[tuple[int, int], float]):
    autogen_config = concept.class_set.auto_generate

    class_images_dir = Path(concept.class_set.path)
    class_images_dir.mkdir(parents=True, exist_ok=True)

    cur_class_images = list(SDDataset.get_images(class_images_dir))
    class_id_size_map = SDDatasetWithARB.get_id_size_map(cur_class_images)

    cur_dist = {size: len([k for k, v in class_id_size_map.items() if v == size]) for id, size in
                class_id_size_map.items()}
    target_dist = {size: int(autogen_config.num_target * p) for size, p in size_dist.items()}

    dist_diff = {k: v - cur_dist.get(k, 0) for k, v in target_dist.items() if v > cur_dist.get(k, 0)}

    num_new_images = sum(dist_diff.values())
    logger.info(f"Total number of class images to sample: {num_new_images}.")

    with torch.autocast("cuda"), torch.inference_mode(), \
            tqdm(total=num_new_images, desc="Generating class images") as progress:
        for (w, h), target in dist_diff.items():
            progress.set_postfix({"size": (w, h)})

            while True:
                actual_bs = target if target - autogen_config.batch_size < 0 \
                    else autogen_config.batch_size

                if actual_bs <= 0:
                    break

                images = pipeline(
                    prompt=concept.class_set.prompt,
                    negative_prompt=autogen_config.negative_prompt,
                    guidance_scale=autogen_config.cfg_scale,
                    num_inference_steps=autogen_config.steps,
                    num_images_per_prompt=actual_bs,
                    width=w,
                    height=h
                ).images

                for image in images:
                    hash = hashlib.md5(image.tobytes()).hexdigest()
                    image_filename = class_images_dir / f"{hash}.jpg"
                    image.save(image_filename)
                    progress.update()

                target -= actual_bs


@click.command()
@click.option("--config", required=True)
def main(config):
    default_config = OmegaConf.load('configs/__reserved_default__.yaml')
    config = OmegaConf.merge(default_config, OmegaConf.load(config))

    if not config.prior_preservation.enabled:
        logger.warning("Prior preservation not enabled. Class image generation is not needed.")
        return

    pipeline = StableDiffusionPipeline.from_pretrained(config.model)
    pipeline.set_progress_bar_config(disable=True)
    pipeline.to("cuda")
    pipeline.unet.half()
    pipeline.unet.set_use_memory_efficient_attention_xformers(True)

    for i, concept in enumerate(config.data.concepts):
        if not concept.class_set.auto_generate.enabled:
            logger.warning(f"Concept [{i}] skipped because class auto generate is not enabled.")
            continue

        if not config.aspect_ratio_bucket.enabled:
            size_dist = {(config.data.resolution, config.data.resolution): 1.0}
        else:
            instance_img_paths = list(SDDataset.get_images(Path(concept.instance_set.path)))
            id_size_map = SDDatasetWithARB.get_id_size_map(instance_img_paths)
            bucket_manager = BucketManager(114514, 1919810, 69, 418)
            bucket_manager.gen_buckets()
            bucket_manager.put_in(id_size_map)
            size_dist = {bucket.size: len(bucket.ids) / len(instance_img_paths) for bucket in
                         bucket_manager.buckets}
            assert sum(size_dist.values()) == 1

        generate_class_images(pipeline, concept, size_dist)


if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    main()
