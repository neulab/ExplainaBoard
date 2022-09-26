"""Tests for explainaboard.processors.text_pair_classification"""

from __future__ import annotations

import unittest

from explainaboard import TaskType
from explainaboard.processors.processor_factory import get_processor
from explainaboard.processors.text_pair_classification import (
    TextPairClassificationProcessor,
)
from explainaboard.serialization.serializers import PrimitiveSerializer


class TextPairClassificationProcessorTest(unittest.TestCase):
    def test_get_processor(self) -> None:
        self.assertIsInstance(
            get_processor(TaskType.text_pair_classification.value),
            TextPairClassificationProcessor,
        )

    def test_serialize(self) -> None:
        serializer = PrimitiveSerializer()
        self.assertEqual(
            serializer.serialize(TextPairClassificationProcessor()),
            {"cls_name": "TextPairClassificationProcessor"},
        )

    def test_deserialize(self) -> None:
        serializer = PrimitiveSerializer()
        self.assertIsInstance(
            serializer.deserialize({"cls_name": "TextPairClassificationProcessor"}),
            TextPairClassificationProcessor,
        )
