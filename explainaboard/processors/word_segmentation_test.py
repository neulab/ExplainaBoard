"""Tests for explainaboard.processors.word_segmentation"""

from __future__ import annotations

import unittest

from explainaboard.processors.word_segmentation import CWSProcessor
from explainaboard.serialization.serializers import PrimitiveSerializer


class CWSProcessorTest(unittest.TestCase):
    def test_serialize(self) -> None:
        serializer = PrimitiveSerializer()
        self.assertEqual(
            serializer.serialize(CWSProcessor()),
            {"cls_name": "CWSProcessor"},
        )

    def test_deserialize(self) -> None:
        serializer = PrimitiveSerializer()
        self.assertIsInstance(
            serializer.deserialize({"cls_name": "CWSProcessor"}),
            CWSProcessor,
        )
