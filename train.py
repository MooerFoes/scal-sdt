import logging
from argparse import Namespace
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint

from modules.args import parser
from modules.model import load_model
from modules.sample_callback import SampleCallback

logger = logging.getLogger()


def get_resuming_config(ckpt_path: Path) -> Optional[DictConfig]:
    config_yaml = ckpt_path.parent / "config.yaml"
    if not config_yaml.is_file():
        return None

    return OmegaConf.load(config_yaml)


def generate_run_id() -> str:
    import time
    return time.strftime("%y%m%d-%H%M%S")


def get_params():
    args = parser.parse_args()

    config = OmegaConf.load('configs/__reserved_default__.yaml')

    assert args.resume or args.config, "Either resume or config must be specified"

    if args.resume:
        logger.info("Trying to resume training, loading config from checkpoint")
        resume_config = get_resuming_config(Path(args.resume))
        if resume_config:
            config = OmegaConf.merge(config, resume_config)
        else:
            logger.warning("Config not found for the checkpoint specified")

    if args.config:
        config = OmegaConf.merge(config, OmegaConf.load(args.config))

    if args.run_id is None:
        args.run_id = generate_run_id()

    return args, config


def verify_config(config: DictConfig):
    concepts = config.data.concepts
    assert any(concepts)

    if not config.prior_preservation.enabled:
        logger.info("Running: Standard Finetuning")
        if any(concept for concept in concepts if concept.get("class_set") is not None):
            logger.warning("Prior preservation loss is disabled, but there's concept with class set specified")
    else:
        logger.info("Running: DreamBooth")
        assert all(concept.get("class_set") is not None for concept in concepts), \
            "Prior preservation loss is enabled, but not all concepts have class set specified"


def get_loggers(config: DictConfig):
    project_dir = Path(config.output_dir, config.project)

    train_loggers = list[pl.loggers.Logger]()
    if config.loggers.get("tensorboard") is not None:
        from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
        train_loggers.append(TensorBoardLogger(save_dir=str(project_dir)))

    if config.loggers.get("wandb") is not None:
        from pytorch_lightning.loggers.wandb import WandbLogger
        train_loggers.append(WandbLogger(project=config.project, save_dir=str(project_dir)))

    return train_loggers


def do_disable_amp_hack(model, config, trainer):
    match config.trainer.precision:
        case 16:
            model.unet = model.unet.to(torch.float16)
        case "bf16":
            model.unet = model.unet.to(torch.bfloat16)

    # Dirty hack to silent "Attempting to unscale FP16 gradients"
    from pytorch_lightning.plugins import PrecisionPlugin
    precision_plugin = PrecisionPlugin()
    precision_plugin.precision = config.trainer.precision
    trainer.strategy.precision_plugin = precision_plugin


def main(args: Namespace, config: DictConfig):
    verify_config(config)

    ckpt_save_dir = Path(config.output_dir, config.project, args.run_id)
    ckpt_save_dir.mkdir(parents=True, exist_ok=True)

    if config.seed is not None:
        pl.seed_everything(config.seed)

    model = load_model(config)

    # TODO
    # if not ("augment" in config.data and any(config.data.augment)):
    # if config.train_text_encoder:
    #     train_dataset.do_cache(model.vae)
    # else:
    #     train_dataset.do_cache(model.vae, model.text_encoder)

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

    if config.force_disable_amp:
        logger.info("Using direct cast, forcibly disabling AMP")
        do_disable_amp_hack(model, config, trainer)

    if args.resume is None:
        trainer.tune(model=model)
    else:
        logger.info("Resuming, will not tune hyperparams")

    OmegaConf.save(config, ckpt_save_dir / "config.yaml")

    trainer.fit(model=model, ckpt_path=args.resume)


if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    main(*get_params())
