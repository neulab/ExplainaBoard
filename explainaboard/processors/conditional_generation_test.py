"""Tests for explainaboard.processors.conditional_generation"""

from __future__ import annotations

import unittest

from explainaboard.processors.conditional_generation import (
    ConditionalGenerationProcessor,
)
from explainaboard.serialization.serializers import PrimitiveSerializer


class ConditionalGenerationProcessorTest(unittest.TestCase):
    def test_serialize(self) -> None:
        serializer = PrimitiveSerializer()
        self.assertEqual(
            serializer.serialize(ConditionalGenerationProcessor()),
            {"cls_name": "ConditionalGenerationProcessor"},
        )

    def test_deserialize(self) -> None:
        serializer = PrimitiveSerializer()
        self.assertIsInstance(
            serializer.deserialize({"cls_name": "ConditionalGenerationProcessor"}),
            ConditionalGenerationProcessor,
        )
