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
    default=False,
    action="store_true",
    help="""Whether to resume from the config and the run id.
If not, load optimizer state etc will not be loaded."""
)

pl.Trainer.add_argparse_args(parser)
