"""Tests for explainaboard.processors.tabular_classification"""

from __future__ import annotations

import unittest

from explainaboard import TaskType
from explainaboard.processors.processor_factory import get_processor
from explainaboard.processors.tabular_classification import (
    TabularClassificationProcessor,
)
from explainaboard.serialization.serializers import PrimitiveSerializer


class TabularClassificationProcessorTest(unittest.TestCase):
    def test_get_processor(self) -> None:
        self.assertIsInstance(
            get_processor(TaskType.tabular_classification.value),
            TabularClassificationProcessor,
        )

    def test_serialize(self) -> None:
        serializer = PrimitiveSerializer()
        self.assertEqual(
            serializer.serialize(TabularClassificationProcessor()),
            {"cls_name": "TabularClassificationProcessor"},
        )

    def test_deserialize(self) -> None:
        serializer = PrimitiveSerializer()
        self.assertIsInstance(
            serializer.deserialize({"cls_name": "TabularClassificationProcessor"}),
            TabularClassificationProcessor,
        )
