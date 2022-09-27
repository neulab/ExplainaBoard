"""Tests for explainaboard.processors.machine_translation"""

from __future__ import annotations

import unittest

from explainaboard.processors.machine_translation import MachineTranslationProcessor
from explainaboard.serialization.serializers import PrimitiveSerializer


class MachineTranslationProcessorTest(unittest.TestCase):
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
