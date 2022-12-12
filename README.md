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

Linux is recommended, on Windows you have to install `bitsandbytes` manually for int8 optimizers.

#### Conda

```shell
conda env create environment.yml
conda activate ssdt
```

#### Pip

CUDA toolkit and torch should be installed manually.

```shell
pip install -r requirements.txt
```

### Config

**Documentation**: `configs/README.md`.
([Link](https://github.com/CCRcmcpe/scal-sdt/blob/main/configs/README.md))

`configs/native.yaml` (for native training) and `configs/dreambooth.yaml` (for DreamBooth) provided as examples.

### Run

If you are running native training, proceed to the next step.  
If you are running DreamBooth, run this to generate class (regularization) images:

```shell
python gen_class_imgs.py --config configs/your_config.yaml
```

Then run the training:

```shell
python train.py --config configs/your_config.yaml
```

### After Training

Note although the checkpoints have `.ckpt` extension, they are NOT directly usable to interfaces based on
the [official SD code base](https://github.com/CompVis/stable-diffusion)
like [WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui). To convert them into SD checkpoints:

```shell
python convert_to_sd.py PATH_TO_THE_CKPT OUTPUTDIR --no-text-encoder --unet-dtype fp16
```

`--no-text-encoder --unet-dtype fp16` results a ~2GB checkpoint, containing fp16 UNet and fp32 VAE weights, WebUI
supports loading that. For further reducing checkpoint size to ~1.6GB if target clients have external VAE already,
add `--no-vae` to remove VAE weights from checkpoint, leaving fp16 UNet weights only.

If you are not using WebUI and having issues, remove `--no-text-encoder`.

### TPUs or other computing units?

You may change `trainer.accelerator`.
([Docs](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#pytorch_lightning.trainer.Trainer.params.accelerator))

### Advanced

Check out the [wiki](https://github.com/CCRcmcpe/scal-sdt/wiki). Contains some information for training efficiently.
