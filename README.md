# SCAL-SDT

Scalable Stable Diffusion Trainer

[!] IN EARLY DEVELOPMENT, CONFIGS SUBJECT TO BREAKING CHANGES

## Features

* Can run with <8GB VRAM
* [Aspect Ratio Bucketing](https://github.com/NovelAI/novelai-aspect-ratio-bucketing)
* CLIP skip
* WandB logging

Customizable training objective, including:

* [DreamBooth](https://arxiv.org/abs/2208.12242)
* [Custom Diffusion](https://github.com/adobe-research/custom-diffusion) (Partial)
* [Low-Rank Adaptation (LoRA)](https://arxiv.org/abs/2106.09685)

## Getting Started

### Install Requirements

Linux is recommended, on Windows you have to install `bitsandbytes` manually for int8 optimizers.

#### Conda

```shell
conda env create -f environment.yml
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

In `configs`, `native.yaml` (for so-called native training), `dreambooth.yaml`, `lora.yaml` provided as examples.

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

[WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
may not directly load the checkpoints due to the safe loading mechanism.
To solve this issue and reduce checkpoint size:

```shell
python ckpt_tool.py prune INPUT_CKPT OUTPUT_CKPT --unet-dtype fp16
```

Results a ~1.6GB checkpoint which can be loaded by WebUI, containing fp16 UNet.

If you are not using WebUI and having issues, add `--text-encoder` and `--vae`
and remove `--unet-dtype fp16` to get a full checkpoint.

### TPUs or other computing units?

You may change `trainer.accelerator`.
([Docs](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#pytorch_lightning.trainer.Trainer.params.accelerator))

### Advanced

* [Config Documentation](https://github.com/CCRcmcpe/scal-sdt/blob/main/configs/README.md)
* [SCAL-SDT Wiki](https://github.com/CCRcmcpe/scal-sdt/wiki)
