import logging
import sys

from pythonjsonlogger import json


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Shared JSON logger with no gateway dependency."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            json.JsonFormatter(
                "%(asctime)s %(levelname)s %(name)s %(process)d %(message)s"
            )
        )
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger


triton_logger = setup_logger("triton")
