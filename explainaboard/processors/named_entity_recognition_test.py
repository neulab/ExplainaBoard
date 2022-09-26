"""Tests for explainaboard.processors.named_entity_recognition"""

from __future__ import annotations

import unittest

from explainaboard import TaskType
from explainaboard.processors.named_entity_recognition import NERProcessor
from explainaboard.processors.processor_factory import get_processor
from explainaboard.serialization.serializers import PrimitiveSerializer


class NERProcessorTest(unittest.TestCase):
    def test_get_processor(self) -> None:
        self.assertIsInstance(
            get_processor(TaskType.named_entity_recognition.value), NERProcessor
        )

    def test_serialize(self) -> None:
        serializer = PrimitiveSerializer()
        self.assertEqual(
            serializer.serialize(NERProcessor()),
            {"cls_name": "NERProcessor"},
        )

    def test_deserialize(self) -> None:
        serializer = PrimitiveSerializer()
        self.assertIsInstance(
            serializer.deserialize({"cls_name": "NERProcessor"}),
            NERProcessor,
        )
