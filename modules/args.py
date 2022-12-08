import argparse

import pytorch_lightning as pl

parser = argparse.ArgumentParser(
    description="SCAL-SDT, the Scalable Stable Diffusion Trainer."
)
parser.add_argument(
    "--config",
    type=str,
    default=None,
    help="Path to the training config file."
)
parser.add_argument(
    "--run-id",
    type=str,
    default=None,
    help="Id of this run for saving checkpoint, defaults to current time formatted to yyddmm-HHMMSS."
)
parser.add_argument(
    "--resume",
    type=str,
    default=None,
    help="Resume from the specified checkpoint path. Corresponding config will be loaded if exists."
)

pl.Trainer.add_argparse_args(parser)
