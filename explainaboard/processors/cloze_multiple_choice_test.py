"""Tests for explainaboard.processors.cloze_multiple_choice"""

from __future__ import annotations

import unittest

from explainaboard import TaskType
from explainaboard.processors.cloze_multiple_choice import ClozeMultipleChoiceProcessor
from explainaboard.processors.processor_factory import get_processor
from explainaboard.serialization.serializers import PrimitiveSerializer


class ClozeMultipleChoiceProcessorTest(unittest.TestCase):
    def test_get_processor(self) -> None:
        self.assertIsInstance(
            get_processor(TaskType.cloze_mutiple_choice.value),
            ClozeMultipleChoiceProcessor,
        )

    def test_serialize(self) -> None:
        serializer = PrimitiveSerializer()
        self.assertEqual(
            serializer.serialize(ClozeMultipleChoiceProcessor()),
            {"cls_name": "ClozeMultipleChoiceProcessor"},
        )

    def test_deserialize(self) -> None:
        serializer = PrimitiveSerializer()
        self.assertIsInstance(
            serializer.deserialize({"cls_name": "ClozeMultipleChoiceProcessor"}),
            ClozeMultipleChoiceProcessor,
        )
