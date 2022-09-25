"""Tests for explainaboard.processors.qa_tat"""

from __future__ import annotations

import unittest

from explainaboard import TaskType
from explainaboard.processors.processor_registry import get_processor
from explainaboard.processors.qa_tat import QATatProcessor
from explainaboard.serialization.serializers import PrimitiveSerializer


class QATatProcessorTest(unittest.TestCase):
    def test_get_processor(self) -> None:
        self.assertIsInstance(
            get_processor(TaskType.qa_tat.value),
            QATatProcessor,
        )

    def test_serialize(self) -> None:
        serializer = PrimitiveSerializer()
        self.assertEqual(
            serializer.serialize(QATatProcessor()),
            {"cls_name": "QATatProcessor"},
        )

    def test_deserialize(self) -> None:
        serializer = PrimitiveSerializer()
        self.assertIsInstance(
            serializer.deserialize({"cls_name": "QATatProcessor"}),
            QATatProcessor,
        )
