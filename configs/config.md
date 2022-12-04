# Config

## Notable Entries

`model`: Only Diffusers format is currently supported. Can be path to saved diffusers SD pipeline or a HuggingFace model
identifier.

`checkpoint.every_n_epochs`: Determines how frequent checkpoints should be saved. `checkpoint.save_top_k` also
determines how many latest checkpoints should be kept. If your storage is limited, consider changing those.

`loggers.wandb`: If you don't want to use WandB, you should remove this entry.

## Data Augmentation

Example:

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
Can be one
of `torchvision.transforms`([Reference](https://pytorch.org/vision/stable/transforms.html#transforms-on-pil-image-and-torch-tensor)).
SCAL-SDT itself provides `modules.dataset.augment.RandomRotationWithCrop` for rotate and crop.

`params`: Parameters for instantiating the class.

Augmentation transforms are applied before normalization on data.
