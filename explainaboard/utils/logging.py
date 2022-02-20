import logging
from typing import Optional

log_levels = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

_default_log_level = logging.WARNING


def _get_library_name() -> str:
    return __name__.split(".")[0]


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger with the specified name.
    This function can be used in dataset and metrics scripts.
    """
    if name is None:
        name = _get_library_name()
    return logging.getLogger(name)
