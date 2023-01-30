# SCAL-SDT

Scalable Stable Diffusion Trainer

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

#### PyPI

Python 3.10 is required. CUDA toolkit and torch should be installed manually.

```shell
pip install -r requirements.txt
```

### Config

**Documentation**: `configs/README.md`.
([Link](https://github.com/CCRcmcpe/scal-sdt/blob/main/configs/README.md))

In `configs` directory, `native.yaml` (for so-called native training), `dreambooth.yaml`, `lora.yaml` provided as examples.

### Run

```shell
python train.py --config configs/your_config.yaml
```

But if you are running DreamBooth, run this first to generate regularization images:

```shell
python gen_class_imgs.py --config configs/your_config.yaml
```

### After Training

[WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
may not directly load the checkpoints due to the safe loading mechanism.
To solve this issue and reduce checkpoint size:

```shell
python ckpt_tool.py prune INPUT OUTPUT --unet-dtype fp16
```

`INPUT` is the path to the trained SCAL-SDT checkpoint.

If `OUTPUT` has suffix `.safetensors` then safetensors format will be used.

A ~1.6GB file will be created, which can be loaded by WebUI, containing fp16 UNet states.

If you are not using WebUI and having issues, specify both `--text-encoder` and `--vae`
and remove `--unet-dtype fp16` to get a full checkpoint.

### Advanced

* [Config Documentation](https://github.com/CCRcmcpe/scal-sdt/blob/main/configs/README.md)
* [SCAL-SDT Wiki](https://github.com/CCRcmcpe/scal-sdt/wiki)
