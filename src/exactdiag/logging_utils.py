import atexit
import datetime as dt
import json
import logging
import os
import pathlib
from time import time
from typing import override

import numpy as np

# This package is shamelessly copied from an mCoding video.


class JSON_Encoder(json.JSONEncoder):
    def default(self, obj):
        match obj:
            case np.generic():
                return obj.item()
            case np.ndarray():
                return list(obj.tolist())
        super().default(obj)


def timeit(func):
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        end = time()
        logging.getLogger(__name__).debug(f"Time elapsed in {func.__name__}: {end - start} s.")
        return result

    return wrapper


def setup_logging() -> None:
    """Set up a basic logging to stdout, stderr, and `logs/` folder."""
    config_file = pathlib.Path(__file__).parent / "logging_config.json"
    with open(config_file, mode="r", encoding="UTF-8") as fp:
        config = json.load(fp)
    os.makedirs("logs", exist_ok=True)
    logging.config.dictConfig(config)
    queue_handler = logging.getHandlerByName("queue_handler")
    if queue_handler is not None:
        queue_handler.listener.start()
        atexit.register(queue_handler.listener.stop)


class Below_Warning_Filter(logging.Filter):
    @override
    def filter(self, record: logging.LogRecord) -> bool | logging.LogRecord:
        return record.levelno <= logging.INFO


class This_Package_Filter(logging.Filter):
    @override
    def filter(self, record: logging.LogRecord) -> bool | logging.LogRecord:
        package = "exactdiag"
        return record.name[: len(package)] == package


class JSON_Formatter(logging.Formatter):
    def __init__(self, *, fmt_keys: dict[str, str] | None = None):
        super().__init__()
        self.fmt_keys = fmt_keys if fmt_keys is not None else {}

    @override
    def format(self, record: logging.LogRecord) -> str:
        message = self._prepare_log_dict(record)
        return json.dumps(message, default=str)

    def _prepare_log_dict(self, record: logging.LogRecord):
        always_fields = {
            "message": record.getMessage(),
            "timestamp": dt.datetime.fromtimestamp(record.created, tz=dt.timezone.utc).isoformat(),
        }
        if record.exc_info is not None:
            always_fields["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info is not None:
            always_fields["stack_info"] = self.formatException(record.stack_info)

        message = {
            key: msg_val if (msg_val := always_fields.pop(val, None)) is not None else getattr(record, val)
            for key, val in self.fmt_keys.items()
        }
        message.update(always_fields)
        return message
