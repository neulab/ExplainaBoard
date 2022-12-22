"""Tests for py_utils.py."""

from __future__ import annotations

import math
import unittest

from explainaboard.utils.py_utils import replace_nan


class ReplaceNanTest(unittest.TestCase):
    def test_replace_nan(self) -> None:
        self.assertEqual(replace_nan(math.nan, 10.0), 10.0)

        self.assertEqual(replace_nan(1.0, 10.0), 1.0)
