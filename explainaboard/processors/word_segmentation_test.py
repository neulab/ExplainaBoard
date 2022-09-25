"""Tests for explainaboard.processors.word_segmentation"""

from __future__ import annotations

import unittest

from explainaboard import TaskType
from explainaboard.processors.processor_registry import get_processor
from explainaboard.processors.word_segmentation import CWSProcessor
from explainaboard.serialization.serializers import PrimitiveSerializer


class CWSProcessorTest(unittest.TestCase):
    def test_get_processor(self) -> None:
        self.assertIsInstance(
            get_processor(TaskType.word_segmentation.value), CWSProcessor
        )

    def test_serialize(self) -> None:
        serializer = PrimitiveSerializer()
        self.assertEqual(
            serializer.serialize(CWSProcessor()),
            {"cls_name": "CWSProcessor"},
        )

    def test_deserialize(self) -> None:
        serializer = PrimitiveSerializer()
        self.assertIsInstance(
            serializer.deserialize({"cls_name": "CWSProcessor"}),
            CWSProcessor,
        )
