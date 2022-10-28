"""Logging utilities."""

from __future__ import annotations

import logging
import os
from typing import Optional, TypeVar

from tqdm import tqdm

hide_progress = "EXPLAINABOARD_HIDE_PROGRESS" in os.environ

T = TypeVar("T")


def progress(gen, desc: str = None):
    """Create a progress bar.

    Args:
        gen: An iterator
        desc: The description on the progress bar

    Returns:
        An iterator with a progress bar unless hide_progress is on.
    """
    return gen if hide_progress else tqdm(gen, desc=desc)


def _get_library_name() -> str:
    """Helper to obtain the library name."""
    return __name__.split(".")[0]


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger with the specified name.

    This function can be used in dataset and metrics scripts.

    Args:
        name: Name of the logger.

    Returns:
        Logger object associated to `name`.
    """
    full_name = _get_library_name()
    if name is not None:
        full_name = f"{full_name}.{name}"
    return logging.getLogger(full_name)
