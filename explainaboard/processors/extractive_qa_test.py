"""Tests for explainaboard.processors.extractive_qa"""

from __future__ import annotations

import unittest

from explainaboard.processors.extractive_qa import QAExtractiveProcessor
from explainaboard.serialization.serializers import PrimitiveSerializer


class QAExtractiveProcessorTest(unittest.TestCase):
    def test_serialize(self) -> None:
        serializer = PrimitiveSerializer()
        self.assertEqual(
            serializer.serialize(QAExtractiveProcessor()),
            {"cls_name": "QAExtractiveProcessor"},
        )

    def test_deserialize(self) -> None:
        serializer = PrimitiveSerializer()
        self.assertIsInstance(
            serializer.deserialize({"cls_name": "QAExtractiveProcessor"}),
            QAExtractiveProcessor,
        )
