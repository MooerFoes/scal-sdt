# Config

## Remarks

SCAL-SDT exposes a lot of implementation details by design to maximize customizability.

PyTorch Lightning is used as underlying framework, checking its docs will be helpful.

Link to docs of PyTorch Lightning will be mentioned below.

Defaults are at `configs/__reserved_default__.yaml`.

## Important Entries

`model`: Path to a saved diffusers SD pipeline or a HuggingFace model identifier.
Only Diffusers format is currently supported, i.e. you cannot directly put a original SD checkpoint.

`loggers.wandb`: If you don't want to use WandB, you should remove this entry.

## Trainer

Config section: `trainer`. Some notable parameters:

`precision`: Default is `16`, which means use FP16 Automatic Mixed Precision (AMP) when training.
You can change it to `'bf16'` if you are using Ampere or other device architecture that supports bfloat16,
as bfloat16 is more suitable for training.

`accumulate_grad_batches`: To do gradient accumulation, set this to the number of batches you want to accumulate.
Can be used to achieve higher batch size than your memory can fit. For example if you set it to `x`, and `batch_size`
was
set to `k`, the effective batch size will be `x * k`.

`auto_scale_batch_size`: If set to one of `power`, `binsearch`,
will automatically find the maximum batch size that can fit into memory.

Also see:
[Trainer API](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-class-api)

## Checkpointing

Config section: `checkpoint`. Some notable parameters:

`every_n_epochs`: Determines how frequent checkpoints should be saved, in epoch unit.

`save_top_k`: Determines how many latest checkpoints should be kept. Old checkpoints will be removed.
If set to `-1`, all checkpoints are kept.

If your storage is limited, consider changing `save_top_k` and `every_n_epochs`.

Also see:
[Checkpointing API](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.ModelCheckpoint.html)

## Optimizer

Config section: `optimizer`.

`name`: The default optimizer is `bitsandbytes.optim.AdamW8bit`,
you can change it to a full precision optimizer like `torch.optim.AdamW`, or `torch.optim.SGD`,
or others, like deepspeed optimizers.

`params`: The parameters passed to the optimizer. Most important is `lr`, the initial learning rate.

`lr_scale`: As effective batch size increases, LR should be adjusted accordingly.
SCAL-SDT has the ability to do this automatically.

| `method` | Formula                         |
|----------|---------------------------------|
| `sqrt`   | `lr = lr * sqrt(effective_bsz)` |
| `linear` | `lr = lr * effective_bsz`       |

`lr_scheduler`: Most of the time you want to use the default LR scheduler (cosine annealing).
`T_max` should be identical to `trainer.max_epochs`.

Refer to [PyTorch docs](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
for more LR schedulers.

## Data Augmentation

Config section: `augment`. Example:

```yaml
# At the root of config hierarchy
augment:
  - name: 'torchvision.transforms.RandomHorizontalFlip'
  - name: 'torchvision.transforms.ColorJitter'
    params:
      brightness: 0.05
      contrast: 0.05
      saturation: 0.05
      hue: 0
  - name: 'modules.dataset.augment.RandomRotationWithCrop'
    params:
      angle_deg: 5
      interpolation: bilinear
```

Define a sequence of transforms for augmenting data.

`name`: Full qualifier of a transformer class.
Can be from `torchvision.transforms`
([Reference](https://pytorch.org/vision/stable/transforms.html#transforms-on-pil-image-and-torch-tensor)).
SCAL-SDT itself provides `modules.dataset.augment.RandomRotationWithCrop` for rotate and crop.
The class needs to have `__call__(torch.Tensor) -> torch.Tensor` implemented.

`params`: Parameters for instantiating the class.

Augmentation transforms are applied before normalization.
