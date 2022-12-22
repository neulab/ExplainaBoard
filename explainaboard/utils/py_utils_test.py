"""Tests for py_utils.py."""

from __future__ import annotations

import unittest
import math
from explainaboard.utils.py_utils import replace_nan


class ReplaceNanTest(unittest.TestCase):
    def test_replace_nan(self):
        self.assertEqual(replace_nan(math.nan, 10), 10)

        self.assertEqual(replace_nan(1, 10), 1)
