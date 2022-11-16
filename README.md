# SCAL-SDT

Stable Diffusion trainer with scalable dataset size, scalable hardware requirement and DreamBooth functionality.

### Features

* Can run with 12G or less VRAM without losing speed with prior preservation loss enabled
* Additionally can use deepspeed to further reduce VRAM usage
* [Aspect Ratio Bucketing](https://github.com/NovelAI/novelai-aspect-ratio-bucketing)
* Support CLIP skip
* Support wandb logging
* Support per-image labels (both instance and class set)
* Deepdanbooru labeling script
* Cosine annealing LR scheduler
* You can use it without the dreambooth part (equivalent to standard finetuning process)
