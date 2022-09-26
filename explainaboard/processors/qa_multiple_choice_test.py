"""Tests for explainaboard.processors.qa_multiple_choice"""

from __future__ import annotations

import unittest

from explainaboard import TaskType
from explainaboard.processors.processor_factory import get_processor
from explainaboard.processors.qa_multiple_choice import QAMultipleChoiceProcessor
from explainaboard.serialization.serializers import PrimitiveSerializer


class QAMultipleChoiceProcessorTest(unittest.TestCase):
    def test_get_processor(self) -> None:
        self.assertIsInstance(
            get_processor(TaskType.qa_multiple_choice.value),
            QAMultipleChoiceProcessor,
        )

    def test_serialize(self) -> None:
        serializer = PrimitiveSerializer()
        self.assertEqual(
            serializer.serialize(QAMultipleChoiceProcessor()),
            {"cls_name": "QAMultipleChoiceProcessor"},
        )

    def test_deserialize(self) -> None:
        serializer = PrimitiveSerializer()
        self.assertIsInstance(
            serializer.deserialize({"cls_name": "QAMultipleChoiceProcessor"}),
            QAMultipleChoiceProcessor,
        )
