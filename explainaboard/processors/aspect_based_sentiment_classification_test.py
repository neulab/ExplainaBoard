"""Tests for explainaboard.processors.aspect_based_sentiment_classification"""

from __future__ import annotations

import unittest

from explainaboard.processors.aspect_based_sentiment_classification import (
    AspectBasedSentimentClassificationProcessor,
)
from explainaboard.serialization.serializers import PrimitiveSerializer


class AspectBasedSentimentClassificationProcessorTest(unittest.TestCase):
    def test_serialize(self) -> None:
        serializer = PrimitiveSerializer()
        self.assertEqual(
            serializer.serialize(AspectBasedSentimentClassificationProcessor()),
            {"cls_name": "AspectBasedSentimentClassificationProcessor"},
        )

    def test_deserialize(self) -> None:
        serializer = PrimitiveSerializer()
        self.assertIsInstance(
            serializer.deserialize(
                {"cls_name": "AspectBasedSentimentClassificationProcessor"}
            ),
            AspectBasedSentimentClassificationProcessor,
        )
