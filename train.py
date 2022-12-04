import logging
from collections.abc import Iterable
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.utils.checkpoint
import torch.utils.data
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import CLIPTokenizer

from modules.args import parser
from modules.dataset.datasets import Item
from modules.model import load_model
from modules.sample_callback import SampleCallback

logger = logging.getLogger()


def get_resuming_config(config: DictConfig) -> DictConfig | None:
    state_dir = Path(config.model, "state")
    if not (state_dir.is_dir() and
            (state_dir / "state.pt").is_file()):
        logger.warning("Checkpoint has no state")
        return None

    logger.info("Trying to resume training, loading config from checkpoint")

    config_yaml = state_dir / "config.yaml"
    if not config_yaml.is_file():
        logger.warning("Checkpoint has no config")
        return None

    logger.warning("Merging: checkpoint's config -> provided config")
    return OmegaConf.merge(config, OmegaConf.load(config_yaml))


def generate_run_id() -> str:
    import time
    return time.strftime("%y%m%d-%H%M%S")


def get_params():
    args = parser.parse_args()

    config = OmegaConf.load('configs/__reserved_default__.yaml')
    config = OmegaConf.merge(config, OmegaConf.load(args.config))

    if args.run_id is None:
        args.run_id = generate_run_id()

    if args.resume:
        result = get_resuming_config(config)
        if result:
            config = result
        else:
            logger.warning("Not normally resuming checkpoint")

    OmegaConf.set_readonly(config, True)
    return args, config


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


def get_dataset(config, tokenizer: CLIPTokenizer):
    if config.aspect_ratio_bucket.enabled:
        if config.prior_preservation.enabled:
            from modules.dataset.arb_datasets import DBDatasetWithARB
            dataset_type = DBDatasetWithARB
        else:
            from modules.dataset.arb_datasets import SDDatasetWithARB
            dataset_type = SDDatasetWithARB
    else:
        if config.prior_preservation.enabled:
            from modules.dataset.datasets import DBDataset
            dataset_type = DBDataset
        else:
            from modules.dataset.datasets import SDDataset
            dataset_type = SDDataset

    return dataset_type(
        concepts=config.data.concepts,
        tokenizer=tokenizer,
        size=config.data.resolution,
        center_crop=config.data.center_crop,
        pad_tokens=config.pad_tokens,
        batch_size=config.batch_size,
        bucket_config=config.aspect_ratio_bucket,
        seed=config.seed
    )


def verify_config(config):
    concepts = config.data.concepts
    assert any(concepts)

    if not config.prior_preservation.enabled:
        logger.info("Running: Standard Finetuning")
    else:
        logger.info("Running: DreamBooth")


def get_loggers(config):
    project_dir = Path(config.output_dir, config.project)

    train_loggers = list[pl.loggers.Logger]()
    if config.loggers.get("tensorboard") is not None:
        from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
        train_loggers.append(TensorBoardLogger(save_dir=str(project_dir)))

    if config.loggers.get("wandb") is not None:
        from pytorch_lightning.loggers.wandb import WandbLogger
        train_loggers.append(WandbLogger(project=config.project, save_dir=str(project_dir)))

    return train_loggers


def main(args, config):
    verify_config(config)

    ckpt_save_dir = Path(config.output_dir, config.project, args.run_id)
    ckpt_save_dir.mkdir(parents=True, exist_ok=True)

    if config.seed:
        pl.seed_everything(config.seed)

    model = load_model(config)

    train_dataset = get_dataset(config, model.tokenizer)
    # TODO
    # if not ("augment" in config.data and any(config.data.augment)):
    # if config.train_text_encoder:
    #     train_dataset.do_cache(model.vae)
    # else:
    #     train_dataset.do_cache(model.vae, model.text_encoder)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        num_workers=8
    )

    loggers = get_loggers(config)

    trainer: pl.Trainer = pl.Trainer.from_argparse_args(
        args,
        logger=loggers,
        callbacks=[
            ModelCheckpoint(dirpath=ckpt_save_dir, **config.checkpoint),
            SampleCallback(ckpt_save_dir / "samples")
        ],
        benchmark=not config.aspect_ratio_bucket.enabled,
        **config.trainer
    )

    trainer.tune(model=model)
    trainer.fit(
        model=model,
        ckpt_path=ckpt_save_dir if args.resume else None,
        train_dataloaders=train_dataloader
    )


if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    main(*get_params())
