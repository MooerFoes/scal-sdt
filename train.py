import logging
from pathlib import Path
from typing import Optional

import click
import pytorch_lightning as pl
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import StrategyRegistry

from modules import configs
from modules.model import LatentDiffusionModel
from modules.sample_callback import SampleCallback
from modules.utils.fix_ddp import DDPStaticGraphStrategy
from modules.utils.logging import rank_zero_logger

logger: logging.Logger


def get_resuming_config(ckpt_path: Path) -> DictConfig:
    config_yaml = ckpt_path.parent / "config.yaml"
    if not config_yaml.is_file():
        raise FileNotFoundError("Config not found for the checkpoint specified")

    return OmegaConf.load(config_yaml)


def generate_run_id() -> str:
    import time
    return time.strftime("%y%m%d-%H%M%S")


@rank_zero_only
def verify_config(config: DictConfig):
    concepts = config.data.concepts

    have_concepts = any(concepts)

    if have_concepts and config.data.cache is not None:
        logger.warning("One or more concept is set, but won't be used as cache is specified")
    elif not have_concepts:
        raise Exception("No concept found and cache file is not specified")

    if not config.prior_preservation.enabled:
        if any(concept for concept in concepts if concept.get("class_set") is not None):
            logger.warning("Prior preservation loss is disabled, but there's concept with class set specified")
    elif not all(concept.get("class_set") is not None for concept in concepts):
        raise Exception("Prior preservation loss is enabled, but not all concepts have class set specified")


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


@click.command()
@click.option("--config", "config_path",
              type=click.Path(exists=True, dir_okay=False),
              default=None,
              help="Path to the training config file.")
@click.option("--run-id",
              type=str,
              default=None,
              help="Id of this run for saving checkpoint, defaults to current time formatted to yyddmm-HHMMSS.")
@click.option("--resume", "resume_ckpt_path",
              type=click.Path(exists=True, dir_okay=False),
              default=None,
              help="Resume from the specified checkpoint path. Corresponding config will be loaded if exists.")
def main(config_path: Optional[Path],
         run_id: Optional[str],
         resume_ckpt_path: Optional[Path]):
    if config_path is not None:
        config = configs.load_with_defaults(config_path)
    elif resume_ckpt_path is not None:
        config = get_resuming_config(resume_ckpt_path)
    else:
        raise Exception("Either resume or config must be specified")

    if run_id is None:
        run_id = generate_run_id()

    run_dir = Path(config.output_dir, config.project, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    loggers = get_loggers(config)

    StrategyRegistry.register("ddp_static_graph", DDPStaticGraphStrategy, find_unused_parameters=False)

    trainer = pl.Trainer(
        logger=loggers,
        callbacks=[
            ModelCheckpoint(dirpath=run_dir, **config.checkpoint),
            SampleCallback(run_dir / "samples")
        ],
        benchmark=not config.aspect_ratio_bucket.enabled,
        replace_sampler_ddp=not config.aspect_ratio_bucket.enabled,
        **config.trainer
    )

    global logger
    logger = rank_zero_logger()

    verify_config(config)

    logger.info(f"Run ID: {run_id}")

    if config.seed is not None:
        pl.seed_everything(config.seed)

    model = LatentDiffusionModel.from_config(config)

    if config.force_disable_amp:
        logger.info("Using direct cast, forcibly disabling AMP")
        model.disable_amp_hack(model, config, trainer)

    if resume_ckpt_path is None:
        trainer.tune(model=model)
    else:
        logger.info("Resuming, will not tune hyperparams")

    OmegaConf.save(config, run_dir / "config.yaml")

    trainer.fit(model=model, ckpt_path=resume_ckpt_path)


if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    main()
