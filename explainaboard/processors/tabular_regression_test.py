"""Tests for explainaboard.processors.tabular_regression"""

from __future__ import annotations

import unittest

from explainaboard.constants import TaskType
from explainaboard.processors.processor_factory import get_processor_class
from explainaboard.processors.tabular_regression import TabularRegressionProcessor


class TabularRegressionProcessorTest(unittest.TestCase):
    def test_get_processor_class(self) -> None:
        self.assertIs(
            get_processor_class(TaskType.tabular_regression), TabularRegressionProcessor
        )
