"""Tests for explainaboard.processors.tabular_regression"""

from __future__ import annotations

import unittest

from explainaboard import TaskType
from explainaboard.processors.processor_registry import get_processor
from explainaboard.processors.tabular_regression import TabularRegressionProcessor


class TabularRegressionProcessorTest(unittest.TestCase):
    def test_get_processor(self) -> None:
        self.assertIsInstance(
            get_processor(TaskType.tabular_regression.value),
            TabularRegressionProcessor,
        )
