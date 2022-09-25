"""Tests for explainaboard.processors.machine_translation"""

from __future__ import annotations

import unittest

from explainaboard import TaskType
from explainaboard.processors.machine_translation import MachineTranslationProcessor
from explainaboard.processors.processor_registry import get_processor
from explainaboard.serialization.serializers import PrimitiveSerializer


class MachineTranslationProcessorTest(unittest.TestCase):
    def test_get_processor(self) -> None:
        self.assertIsInstance(
            get_processor(TaskType.machine_translation.value),
            MachineTranslationProcessor,
        )

    def test_serialize(self) -> None:
        serializer = PrimitiveSerializer()
        self.assertEqual(
            serializer.serialize(MachineTranslationProcessor()),
            {"cls_name": "MachineTranslationProcessor"},
        )

    def test_deserialize(self) -> None:
        serializer = PrimitiveSerializer()
        self.assertIsInstance(
            serializer.deserialize({"cls_name": "MachineTranslationProcessor"}),
            MachineTranslationProcessor,
        )
