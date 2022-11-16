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

logging.basicConfig(level="INFO")
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


# def generate_class_images(concepts, args, noise_scheduler):
#     pipeline = None
#     for concept in concepts:
#         autogen_config = concept.class_set.auto_generate
#
#         if not autogen_config.enabled:
#             continue
#
#         class_images_dir = Path(concept.class_set.path)
#         class_images_dir.mkdir(parents=True, exist_ok=True)
#
#         cur_class_images = len(list(class_images_dir.iterdir()))
#         if cur_class_images >= autogen_config.num_target:
#             break
#
#         num_new_images = autogen_config.num_target - cur_class_images
#         logger.info(f"Number of class images to sample: {num_new_images}.")
#
#         torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
#         if pipeline is None:
#             pipeline = StableDiffusionPipeline.from_pretrained(
#                 args.pretrained_model_name_or_path,
#                 vae=AutoencoderKL.from_pretrained(
#                     args.pretrained_vae_name_or_path or args.pretrained_model_name_or_path,
#                     subfolder=None if args.pretrained_vae_name_or_path else "vae"
#                 ),
#                 scheduler=noise_scheduler,
#                 torch_dtype=torch_dtype,
#                 safety_checker=None
#             )
#             pipeline.set_progress_bar_config(disable=True)
#             pipeline.to(accelerator.device)
#             pipeline.enable_xformers_memory_efficient_attention()
#
#         sample_dataset = PromptDataset([concept.class_set.prompt, autogen_config.negative_prompt], num_new_images)
#         sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=autogen_config.batch_size)
#
#         sample_dataloader = accelerator.prepare(sample_dataloader)
#
#         with torch.autocast("cuda"), \
#                 torch.inference_mode(), \
#                 tqdm(total=num_new_images,
#                      desc="Generating class images",
#                      disable=not accelerator.is_local_main_process) as progress:
#             for example in sample_dataloader:
#                 images = pipeline(prompt=example["prompt"][0][0],
#                                   negative_prompt=example["prompt"][1][0],
#                                   guidance_scale=autogen_config.cfg_scale,
#                                   num_inference_steps=autogen_config.steps,
#                                   num_images_per_prompt=len(example["prompt"][0])).images
#
#                 for i, image in enumerate(images):
#                     hash_image = hashlib.sha1(image.tobytes()).hexdigest()
#                     image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
#                     image.save(image_filename)
#                     progress.update()
#
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()


def get_collate_fn(tokenizer: CLIPTokenizer):
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

        token_ids = tokenizer.pad({"input_ids": token_ids_array}, padding=True, return_tensors="pt").input_ids

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

    return collate_fn


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
        debug=config.aspect_ratio_bucket.debug,
        seed=config.seed
    )


def verify_config(config):
    concepts = config.data.concepts
    assert any(concepts)

    used_read_txt = all(map(
        lambda c: c.instance_set.combine_prompt_from_txt or
                  c.class_set.combine_prompt_from_txt, concepts))

    if config.prior_preservation.enabled:
        if used_read_txt:
            logger.info("Running: DreamBooth (alternative method)")
        else:
            logger.info("Running: DreamBooth (original paper method)")
    elif used_read_txt:
        logger.info("Running: Standard Finetuning")
    else:
        logger.info("Running: [?]")


def main(args, config):
    verify_config(config)
    ckpt_save_dir = Path(config.output_dir, config.project, args.run_id)

    if config.seed:
        pl.seed_everything(config.seed)

    # TODO
    # if config.prior_preservation.enabled:
    #     generate_class_images(concepts, args, noise_scheduler)
    model = load_model(config)

    train_dataset = get_dataset(config, model.tokenizer)
    checkpoint_callback = ModelCheckpoint(dirpath=ckpt_save_dir, **config.checkpoint)

    # TODO
    # if not ("augment" in config.data and any(config.data.augment)):
    # if config.train_text_encoder:
    #     train_dataset.do_cache(model.vae)
    # else:
    #     train_dataset.do_cache(model.vae, model.text_encoder)

    collate_fn = get_collate_fn(model.tokenizer)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        collate_fn=collate_fn
    )

    # train_logger = WandbLogger(project=config.project)

    trainer = pl.Trainer.from_argparse_args(
        args,
        # logger=train_logger,
        callbacks=[checkpoint_callback],
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
    main(*get_params())
