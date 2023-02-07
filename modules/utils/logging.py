import logging
from typing import Optional

import lightning_utilities.core.rank_zero as rank_zero


def rank_zero_logger(name: Optional[str] = None):
    logger = logging.getLogger(name)

    if getattr(rank_zero.rank_zero_only, "rank", None) is None:
        return logger

    logger._log = rank_zero.rank_zero_only(logger._log)
    return logger
