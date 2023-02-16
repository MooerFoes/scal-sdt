from collections.abc import Iterable
from typing import Any

import torch
from omegaconf import DictConfig

Size = tuple[int, int]

from .samplers import ConstantSizeSampler, AspectSampler, AspectSamplerDB, ConstantSizeSamplerDB
from .datasets import TextConditionalItem, Concept, TextConditionalDataset, PriorPreservationDataset, \
    AspectTextConditionalDataset, \
    TextConditionalItemCached, ItemTypeTextConditional, ItemTypeAll, PriorPreservationItem, TextImageConditionalItem, \
    TextImageConditionalDataset, AspectTextImageConditionalDataset

SSDT_Dataset = (TextConditionalDataset | AspectTextConditionalDataset |
                TextImageConditionalDataset | AspectTextImageConditionalDataset |
                PriorPreservationDataset)


def get_dataset(config: DictConfig, use_cache=True):
    if config.controlnet.enabled:
        dataset_type = \
            AspectTextImageConditionalDataset if config.aspect_ratio_bucket.enabled else TextImageConditionalDataset
    else:
        dataset_type = \
            AspectTextConditionalDataset if config.aspect_ratio_bucket.enabled else TextConditionalDataset

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

    return PriorPreservationDataset(instance_set, class_set)


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


def collate_fn(batch: Iterable[ItemTypeAll]) -> dict[str, Any]:
    ids = list[int]()

    image = list[torch.Tensor]()

    # Conds
    text = list[str]()
    cond_image = list[torch.Tensor]()

    # Cache
    latent = list[torch.Tensor]()
    cond_text = list[torch.Tensor]()

    def append(item: ItemTypeTextConditional | TextImageConditionalItem):
        ids.append(item.id)

        if isinstance(item, TextConditionalItem):
            image.append(item.image)
            text.append(item.text)
        elif isinstance(item, TextConditionalItemCached):
            latent.append(item.latent)
            cond_text.append(item.condition)
        else:
            assert False

        if isinstance(item, TextImageConditionalItem):
            cond_image.append(item.cond_image)

    class_items = []

    for x in batch:
        if isinstance(x, PriorPreservationItem):
            append(x.instance)
            class_items.append(x.prior)
        else:
            append(x)

    for class_item in class_items:
        append(class_item)

    result = {"ids": ids}

    if len(latent) != 0:
        result["latent"] = torch.stack(latent)
        if len(cond_text) != 0:
            result["cond_text"] = torch.stack(cond_text)
    else:
        result["text"] = text
        result["image"] = torch.stack(image)
        if len(cond_image) != 0:
            result["cond_image"] = torch.stack(cond_image)

    return result
