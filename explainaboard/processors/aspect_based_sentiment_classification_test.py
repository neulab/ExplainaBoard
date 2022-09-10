"""Tests for explainaboard.processors.aspect_based_sentiment_classification"""

from __future__ import annotations

import unittest

from explainaboard import TaskType
from explainaboard.processors.aspect_based_sentiment_classification import (
    AspectBasedSentimentClassificationProcessor,
)
from explainaboard.processors.processor_registry import get_processor


class AspectBasedSentimentClassificationProcessorTest(unittest.TestCase):
    def test_get_processor(self) -> None:
        self.assertIsInstance(
            get_processor(TaskType.aspect_based_sentiment_classification.value),
            AspectBasedSentimentClassificationProcessor,
        )
