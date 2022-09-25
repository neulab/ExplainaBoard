"""Tests for explainaboard.processors.tabular_regression"""

from __future__ import annotations

import unittest

from explainaboard import TaskType
from explainaboard.processors.processor_registry import get_processor
from explainaboard.processors.tabular_regression import TabularRegressionProcessor
from explainaboard.serialization.serializers import PrimitiveSerializer


class TabularRegressionProcessorTest(unittest.TestCase):
    def test_get_processor(self) -> None:
        self.assertIsInstance(
            get_processor(TaskType.tabular_regression.value),
            TabularRegressionProcessor,
        )

    def test_serialize(self) -> None:
        serializer = PrimitiveSerializer()
        self.assertEqual(
            serializer.serialize(TabularRegressionProcessor()),
            {"cls_name": "TabularRegressionProcessor"},
        )

    def test_deserialize(self) -> None:
        serializer = PrimitiveSerializer()
        self.assertIsInstance(
            serializer.deserialize({"cls_name": "TabularRegressionProcessor"}),
            TabularRegressionProcessor,
        )
