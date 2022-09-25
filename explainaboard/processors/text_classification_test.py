"""Tests for explainaboard.processors.text_classification"""

from __future__ import annotations

import unittest

from explainaboard import TaskType
from explainaboard.processors.processor_registry import get_processor
from explainaboard.processors.text_classification import TextClassificationProcessor
from explainaboard.serialization.serializers import PrimitiveSerializer


class TextClassificationProcessorTest(unittest.TestCase):
    def test_get_processor(self) -> None:
        self.assertIsInstance(
            get_processor(TaskType.text_classification.value),
            TextClassificationProcessor,
        )

    def test_serialize(self) -> None:
        serializer = PrimitiveSerializer()
        self.assertEqual(
            serializer.serialize(TextClassificationProcessor()),
            {"cls_name": "TextClassificationProcessor"},
        )

    def test_deserialize(self) -> None:
        serializer = PrimitiveSerializer()
        self.assertIsInstance(
            serializer.deserialize({"cls_name": "TextClassificationProcessor"}),
            TextClassificationProcessor,
        )
