from collections.abc import Iterable

import torch
from omegaconf import DictConfig

Size = tuple[int, int]

from .samplers import ConstantSizeSampler, AspectSampler, AspectSamplerDB, ConstantSizeSamplerDB
from .datasets import Item, Concept, ImagePromptDataset, DBDataset, AspectDataset, CacheItem, ItemType

SSDT_Dataset = ImagePromptDataset | AspectDataset | DBDataset


def get_dataset(config: DictConfig, use_cache=True):
    dataset_type = ImagePromptDataset if not config.aspect_ratio_bucket.enabled else AspectDataset
    dataset_params = {
        "center_crop": config.data.center_crop,
        "augment_config": config.get("augment"),
        "cache_file": config.data.cache if use_cache else None
    }

    instance_concepts = [Concept(concept.instance_set.path, concept.instance_set.prompt)
                         for concept in config.data.concepts]
    instance_set = dataset_type(instance_concepts, **dataset_params)

    if not config.prior_preservation.enabled:
        return instance_set

    class_concepts = [Concept(concept.class_set.path, concept.class_set.prompt)
                      for concept in config.data.concepts]
    class_set = dataset_type(class_concepts, **dataset_params)

    return DBDataset(instance_set, class_set)


def get_sampler(dataset: SSDT_Dataset, config: DictConfig, world_size: int, global_rank: int):
    if not config.aspect_ratio_bucket.enabled:
        sampler_type = ConstantSizeSampler if not config.prior_preservation.enabled else ConstantSizeSamplerDB
        return sampler_type(dataset, config.data.resolution)

    arb_params = {
        "data_source": dataset,
        "base_size": config.data.resolution,
        "batch_size": config.batch_size,
        "bucket_config": config.aspect_ratio_bucket,
        "seed": config.seed,
        "world_size": world_size,
        "global_rank": global_rank
    }
    sampler_type = AspectSampler if not config.prior_preservation.enabled else AspectSamplerDB
    return sampler_type(**arb_params)


def collate_fn(batch: Iterable[ItemType | tuple[ItemType, ItemType]]):
    prompt_array = list[str]()
    image_array = list[torch.Tensor]()

    # Cache
    conditions_array = list[torch.Tensor]()
    latents_array = list[torch.Tensor]()

    ids = list[int]()

    class_items = []

    def append(item: ItemType):
        ids.append(item.id)
        if isinstance(item, Item):
            prompt_array.append(item.prompt)
            image_array.append(item.image)
        elif isinstance(item, CacheItem):
            conditions_array.append(item.condition)
            latents_array.append(item.latent)
        else:
            raise Exception()

    for x in batch:
        if isinstance(x, tuple):
            instance_item, class_item = x
            append(instance_item)
            class_items.append(class_item)
        else:
            append(x)

    for class_item in class_items:
        append(class_item)

    result = {"ids": ids}

    if any(latents_array):
        result["latents"] = torch.stack(latents_array)
        if any(conditions_array):
            result["conds"] = torch.stack(conditions_array)
    else:
        result["prompts"] = prompt_array
        result["images"] = torch.stack(image_array)

    return result
