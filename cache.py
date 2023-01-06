import json
import logging
from typing import IO, Optional

import click
import pytorch_lightning as pl
import torch
from diffusers import AutoencoderKL
from safetensors.torch import save_file
from tqdm import trange
from transformers import CLIPTextModel

from modules.config import load_with_defaults
from modules.model import StableDiffusionModel

logger = logging.getLogger("cache")


class CacheBuilder(pl.LightningModule):
    def __init__(self, vae: AutoencoderKL, text_encoder: Optional[CLIPTextModel]):
        super().__init__()
        self.vae = vae
        self.text_encoder = text_encoder

    def batch_all_gather(self, x: torch.Tensor):
        gathered = self.all_gather(x)
        if len(x.shape) == len(gathered.shape):
            return x

        return torch.cat(list(gathered))

    def predict_step(self, batch, batch_idx: int, dataloader_idx=0):
        latents = self.vae.encode(batch["images"]).latent_dist.sample() * 0.18215
        latents = self.batch_all_gather(latents)

        ids = []
        for x in self.all_gather(batch["ids"]):
            if len(x.shape) == 0:
                ids.append(x.item())
            else:
                ids.extend(list(x))

        if self.text_encoder is None:
            return [{"id": id, "latent": latent} for id, latent in zip(ids, latents)]

        conds = self.text_encoder.forward(batch["token_ids"]).last_hidden_state
        conds = self.batch_all_gather(conds)
        return [{"id": id, "latent": latent, "cond": cond} for id, latent, cond in zip(ids, latents, conds)]


@click.command()
@click.option("--config", "config_file",
              type=click.File("r"),
              required=True,
              help="Path to the training config.")
@click.option("--no-conds",
              is_flag=True,
              help="Do not cache condition (useful if training text encoder).")
@click.option("--aug-group-size",
              type=int,
              default=16,
              help="Generate this number of latents per data entry to randomly choose from when training if data augmentation is enabled.")
@click.option("--batch-size",
              type=int,
              default=1,
              help="Batch size, for both text encoder and VAE.")
def main(config_file: IO[str], no_conds: bool, aug_group_size: int, batch_size: int):
    """
    Given training config, generate a dataset cache containing latent and condition (optional) tensors.
    Save path: config entry data.cache

    Limitations:

    * Not data parallel compatible if ARB is enabled.
    * Not compatible with both ARB and augmentation enabled.
    """
    config = load_with_defaults(config_file)
    config.batch_size = batch_size

    if config.data.cache is None:
        raise Exception("data.cache is not set")

    if config.seed is not None:
        pl.seed_everything(config.seed)

    if config.get("augment") is None:
        if aug_group_size != 1:
            logger.warning("Augmentation is not enabled, setting augmentation group size to 1.")
            aug_group_size = 1
    elif config.aspect_ratio_bucket.enabled:
        # As ARB batch entry order is random:
        raise Exception("Caching is not compatible with both Aspect Ratio Bucketing and augmentation enabled.")

    model = StableDiffusionModel.from_config(config)
    cache_builder = CacheBuilder(model.vae, model.text_encoder if not no_conds else None)

    trainer = pl.Trainer(
        benchmark=not config.aspect_ratio_bucket.enabled,
        replace_sampler_ddp=not config.aspect_ratio_bucket.enabled,
        **config.trainer
    )

    model.trainer = trainer
    dataloader = model.train_dataloader(use_cache=False)

    # Release UNet, which is not used.
    del model

    cache_dict = {}
    sizes_info = {}
    ids = set()
    entry_count = 0

    for aug_group_index in trange(aug_group_size):
        # Avoid computing cond more than once per entry.
        if aug_group_index > 1 and not no_conds:
            cache_builder.text_encoder = None

        entry_count = 0
        preds = trainer.predict(cache_builder, dataloader)
        if not trainer.is_global_zero:
            continue

        for batch in preds:
            for result in batch:
                id_ = result['id']
                ids.add(id_)

                latent_key = f"{id_}.latent.{aug_group_index}"
                latent = result["latent"].cpu()
                cache_dict[latent_key] = latent
                sizes_info[latent_key] = latent.shape[1:]

                if "cond" in result:
                    cache_dict[f"{id_}.cond"] = result["cond"].cpu()
                entry_count += 1

    if not trainer.is_global_zero:
        return

    for id_ in ids:
        assert len(set(v.shape for k, v in cache_dict.items() if k.split(".")[:2] == [str(id_), "latent"])) == 1

    logger.info(f"Total entries: {entry_count}")
    logger.info(f"Augmentation group size: {aug_group_size}")

    metadata = {
        "sizes": sizes_info,
        "entries": list(ids),
        "total_entries": entry_count,
        "aug_group_size": aug_group_size
    }

    save_file(cache_dict, config.data.cache, {"json": json.dumps(metadata)})

    logger.info(f'Saved cache to "{config.data.cache}"')


if __name__ == '__main__':
    logging.basicConfig(level="INFO")
    main()
