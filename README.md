# SCAL-SDT

Stable Diffusion trainer with scalable dataset size and hardware usage.

[!] IN EARLY DEVELOPMENT, CONFIGS AND ARGUMENTS SUBJECT TO BREAKING CHANGES

## Features

* Can run with 10G or less VRAM without losing speed thanks to xformers memory efficient attention and int8 optimizers.
* [Aspect Ratio Bucketing](https://github.com/NovelAI/novelai-aspect-ratio-bucketing)
* DreamBooth
* CLIP skip
* WandB logging

## Getting Started

### Install Requirements

Linux is recommended. (If you care to install `bitsandbytes` on Windows)

Python 3.10. Will not work on 3.7.  
Torch 1.13 and CUDA 11.6. Match exact version is recommended but not required.

[xformers](https://github.com/facebookresearch/xformers) is required for efficient VRAM usage. Easiest way to install it is `conda install xformers`.

WandB (`pip install wandb`) is recommended for logging stats and previews.

```sh
pip install -r requirements.txt
```

### Config

Setup a config file first.

`configs/native.yaml` (for native training) and `configs/dreambooth.yaml` (for DreamBooth) provided as examples.

Some notable config entries:

`model`: Only Diffusers format is currently supported. Can be path to saved diffusers SD pipeline or a HuggingFace model identifier.

`checkpoint.every_n_epochs`: This determines how frequent checkpoints should be saved. `checkpoint.save_top_k` also determines how many latest checkpoints should be kept. If your storage is limited, consider changing those.

`loggers.wandb`: If you don't want to use WandB, you should remove this entry.

### Then?

If you are running native training, proceed to the next step.  
If you are running DreamBooth, run this to generate class (regularization) images:

```sh
python gen_class_imgs.py --config configs/your_config.yaml
```

Then run the training:

```sh
python train.py --config configs/your_config.yaml
```

Note although the checkpoints have `.ckpt` extension, they are NOT directly usable to interfaces based on the [official SD code base](https://github.com/CompVis/stable-diffusion) like [WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui). Convert them to SD checkpoints:

```sh
python convert_to_sd.py PATH_TO_THE_CKPT OUTPUTDIR --no-text-encoder --unet-dtype fp16
```

`--no-text-encoder --unet-dtype fp16` results a ~2GB checkpoint, containing fp16 UNet and fp32 VAE weights, WebUI supports loading that. For further reducing checkpoint size to ~1.6GB if target clients have external VAE already, add `--no-vae` to remove VAE weights from checkpoint, leaving fp16 UNet weights only.

If you are not using WebUI and having issues, remove `--no-text-encoder`.

### TPUs or other computing units?

Not well tested, but theoretically supported. You may change `trainer.accelerator`. [Docs about that](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#pytorch_lightning.trainer.Trainer.params.accelerator)

### Advanced

Check out the [wiki](https://github.com/CCRcmcpe/scal-sdt/wiki). Contains some information for training efficiently.
