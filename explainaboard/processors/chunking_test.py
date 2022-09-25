"""Tests for explainaboard.processors.chunking"""

from __future__ import annotations

import unittest

from explainaboard import TaskType
from explainaboard.processors.chunking import ChunkingProcessor
from explainaboard.processors.processor_registry import get_processor
from explainaboard.serialization.serializers import PrimitiveSerializer


class ChunkingProcessorTest(unittest.TestCase):
    def test_get_processor(self) -> None:
        self.assertIsInstance(
            get_processor(TaskType.chunking.value),
            ChunkingProcessor,
        )

    def test_serialize(self) -> None:
        serializer = PrimitiveSerializer()
        self.assertEqual(
            serializer.serialize(ChunkingProcessor()),
            {"cls_name": "ChunkingProcessor"},
        )

    def test_deserialize(self) -> None:
        serializer = PrimitiveSerializer()
        self.assertIsInstance(
            serializer.deserialize({"cls_name": "ChunkingProcessor"}),
            ChunkingProcessor,
        )
