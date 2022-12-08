from collections.abc import Iterable

import torch
from omegaconf import DictConfig
from transformers import CLIPTokenizer

Size = tuple[int, int]

from .samplers import ConstantSizeSampler, AspectSampler, AspectSamplerDB, ConstantSizeSamplerDB
from .datasets import Item, Concept, ImagePromptDataset, DBDataset, AspectDataset

SSDT_Dataset = ImagePromptDataset | AspectDataset | DBDataset


def get_dataset(config: DictConfig, tokenizer: CLIPTokenizer):
    dataset_type = ImagePromptDataset if not config.aspect_ratio_bucket.enabled else AspectDataset
    dataset_params = {
        "tokenizer": tokenizer,
        "center_crop": config.data.center_crop,
        "pad_tokens": config.pad_tokens,
        "augment_config": config.get("augment")
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


def collate_fn(batch: Iterable[Item | tuple[Item, Item]]):
    token_ids_array = list[list[int]]()
    images_array = list[torch.Tensor]()

    # Cache
    # conditions_array = list[torch.Tensor]()
    # latents_array = list[torch.Tensor]()

    class_items = []

    def append(item: Item):
        token_ids_array.append(item.token_ids)
        images_array.append(item.image)
        # conditions_array.append(item.latent)
        # latents_array.append(item.condition)

    for x in batch:
        if isinstance(x, tuple):
            x: tuple[Item, Item]
            instance_item, class_item = x
            append(instance_item)
            class_items.append(class_item)
        else:
            x: Item
            append(x)

    for class_item in class_items:
        append(class_item)

    images = torch.stack(images_array)
    images = images.to(dtype=torch.float32, memory_format=torch.contiguous_format)

    token_ids = torch.tensor(token_ids_array, dtype=torch.int64)

    # conditions = torch.stack(conditions_array)
    #
    # latents = torch.stack(latents_array)

    batch = {
        "token_ids": token_ids,
        "images": images,
        # "conditions": conditions,
        # "latents": latents
    }
    return batch
