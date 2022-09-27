"""Tests for explainaboard.processors.text_pair_classification"""

from __future__ import annotations

import unittest

from explainaboard.processors.text_pair_classification import (
    TextPairClassificationProcessor,
)
from explainaboard.serialization.serializers import PrimitiveSerializer


class TextPairClassificationProcessorTest(unittest.TestCase):
    def test_serialize(self) -> None:
        serializer = PrimitiveSerializer()
        self.assertEqual(
            serializer.serialize(TextPairClassificationProcessor()),
            {"cls_name": "TextPairClassificationProcessor"},
        )

    def test_deserialize(self) -> None:
        serializer = PrimitiveSerializer()
        self.assertIsInstance(
            serializer.deserialize({"cls_name": "TextPairClassificationProcessor"}),
            TextPairClassificationProcessor,
        )
