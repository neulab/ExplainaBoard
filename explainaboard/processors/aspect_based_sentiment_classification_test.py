"""Tests for explainaboard.processors.aspect_based_sentiment_classification"""

from __future__ import annotations

import unittest

from explainaboard import TaskType
from explainaboard.processors.aspect_based_sentiment_classification import (
    AspectBasedSentimentClassificationProcessor,
)
from explainaboard.processors.processor_registry import get_processor
from explainaboard.serialization.serializers import PrimitiveSerializer


class AspectBasedSentimentClassificationProcessorTest(unittest.TestCase):
    def test_get_processor(self) -> None:
        self.assertIsInstance(
            get_processor(TaskType.aspect_based_sentiment_classification.value),
            AspectBasedSentimentClassificationProcessor,
        )

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
