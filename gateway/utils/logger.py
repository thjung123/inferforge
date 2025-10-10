import logging
import sys
from logging.handlers import QueueHandler, QueueListener
from multiprocessing import Queue
from typing import Optional

_log_queue: Optional[Queue] = None
_listener: Optional[QueueListener] = None


class SafeFormatter(logging.Formatter):
    def format(self, record):
        if not hasattr(record, "request_id"):
            record.request_id = "-"
        return super().format(record)


def setup_async_logger(
    name: str = "gateway", level: int = logging.INFO
) -> logging.Logger:
    global _log_queue, _listener

    if _log_queue is None:
        _log_queue = Queue(-1)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(
            SafeFormatter("[%(asctime)s] [%(levelname)s] [%(request_id)s] %(message)s")
        )
        _listener = QueueListener(
            _log_queue, stream_handler, respect_handler_level=True
        )
        _listener.start()

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(QueueHandler(_log_queue))
    logger.propagate = False

    return logger


logger = setup_async_logger()
