"""Tests for explainaboard.loaders.tabular_classification."""

from __future__ import annotations

import unittest

from explainaboard.constants import TaskType
from explainaboard.loaders.loader_factory import get_loader_class
from explainaboard.loaders.tabular_classification import TabularClassificationLoader


class TabularClassificationLoaderTest(unittest.TestCase):
    def test_get_loader_class(self) -> None:
        self.assertIs(
            get_loader_class(TaskType.tabular_classification),
            TabularClassificationLoader,
        )
