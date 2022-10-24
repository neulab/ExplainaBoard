"""Utility functions to manipulate IO."""

from __future__ import annotations

from collections.abc import Generator
import contextlib
import sys
from typing import TextIO


@contextlib.contextmanager
def text_writer(filename: str | None = None) -> Generator[TextIO, None, None]:
    """Prepare a text file to output, or assign STDOUT.

    Args:
        filename: Path to the file, or None to use STDOUT.

    Yields:
        A text file object assigned to the context.
    """
    if filename is None:
        yield sys.stdout
    else:
        with open(filename, "w") as fp:
            yield fp
