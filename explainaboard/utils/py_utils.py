"""Utility functions for basic python libraries."""

from __future__ import annotations

import math


def replace_nan(value: float, default: float) -> float:
    """Replace value with a default value if it is NaN, otherwise, keep it unchanged."""
    return default if math.isnan(value) else value
