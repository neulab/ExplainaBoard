"""Tests for explainaboard.loaders.tabular_regression."""

from __future__ import annotations

import unittest

from explainaboard.constants import TaskType
from explainaboard.loaders.loader_factory import get_loader_class
from explainaboard.loaders.tabular_regression import TabularRegressionLoader


class TabularRegressionLoaderTest(unittest.TestCase):
    def test_get_loader_class(self) -> None:
        self.assertIs(
            get_loader_class(TaskType.tabular_regression), TabularRegressionLoader
        )
