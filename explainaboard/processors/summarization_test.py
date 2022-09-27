"""Tests for explainaboard.processors.summarization"""

from __future__ import annotations

import unittest

from explainaboard.processors.summarization import SummarizationProcessor
from explainaboard.serialization.serializers import PrimitiveSerializer


class SummarizationProcessorTest(unittest.TestCase):
    def test_serialize(self) -> None:
        serializer = PrimitiveSerializer()
        self.assertEqual(
            serializer.serialize(SummarizationProcessor()),
            {"cls_name": "SummarizationProcessor"},
        )

    def test_deserialize(self) -> None:
        serializer = PrimitiveSerializer()
        self.assertIsInstance(
            serializer.deserialize({"cls_name": "SummarizationProcessor"}),
            SummarizationProcessor,
        )
