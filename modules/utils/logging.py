import logging
from types import MethodType
from typing import Optional

import lightning_utilities.core.rank_zero as rank_zero


def rank_zero_logger(name: Optional[str] = None):
    logger = logging.getLogger(name)

    if getattr(rank_zero, "rank", None) is None:
        return logger

    log_rank_zero = rank_zero.rank_zero_only(logger._log)
    logger._log = MethodType(log_rank_zero, logger)
    return logger
