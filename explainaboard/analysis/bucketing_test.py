"""Tests for explainaboard.analysis.bucketing."""

from __future__ import annotations

import unittest

from explainaboard.analysis.bucketing import (
    continuous,
    discrete,
    fixed,
    get_bucketing_method,
)


class ModuleTest(unittest.TestCase):
    def test_get_bucketing_method(self) -> None:
        self.assertIs(get_bucketing_method("continuous"), continuous)
        self.assertIs(get_bucketing_method("discrete"), discrete)
        self.assertIs(get_bucketing_method("fixed"), fixed)

    def test_get_bucketing_method_invalid(self) -> None:
        with self.assertRaisesRegex(
            ValueError, r"^No bucketing method associated to name='xxx'$"
        ):
            get_bucketing_method("xxx")
