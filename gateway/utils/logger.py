import logging
import sys
from pythonjsonlogger import jsonlogger
from gateway.middlewares.request_id import request_id_ctx


class SafeJsonFormatter(jsonlogger.JsonFormatter):
    def process_log_record(self, log_record):
        log_record["request_id"] = request_id_ctx.get() or "-"
        return log_record


def setup_logger(level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger()
    gunicorn_logger = logging.getLogger("gunicorn.access")

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            SafeJsonFormatter(
                "%(asctime)s %(levelname)s %(name)s %(process)d %(request_id)s %(message)s"
            )
        )
        logger.addHandler(handler)
        logger.setLevel(level)

        logger.parent = gunicorn_logger
        logger.propagate = True

    return logger


logger = setup_logger(level=logging.INFO)
