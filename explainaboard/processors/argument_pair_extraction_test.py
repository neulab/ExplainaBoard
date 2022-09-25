"""Tests for explainaboard.processors.argument_pair_extraction"""

from __future__ import annotations

import unittest

from explainaboard import TaskType
from explainaboard.processors.argument_pair_extraction import (
    ArgumentPairExtractionProcessor,
)
from explainaboard.processors.processor_registry import get_processor
from explainaboard.serialization.serializers import PrimitiveSerializer


class ArgumentPairExtractionProcessorTest(unittest.TestCase):
    def test_get_processor(self) -> None:
        self.assertIsInstance(
            get_processor(TaskType.argument_pair_extraction.value),
            ArgumentPairExtractionProcessor,
        )

    def test_serialize(self) -> None:
        serializer = PrimitiveSerializer()
        self.assertEqual(
            serializer.serialize(ArgumentPairExtractionProcessor()),
            {"cls_name": "ArgumentPairExtractionProcessor"},
        )

    def test_deserialize(self) -> None:
        serializer = PrimitiveSerializer()
        self.assertIsInstance(
            serializer.deserialize({"cls_name": "ArgumentPairExtractionProcessor"}),
            ArgumentPairExtractionProcessor,
        )
