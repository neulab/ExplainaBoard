"""Tests for explainaboard.processors.aspect_based_sentiment_classification"""

from __future__ import annotations

import unittest

from explainaboard.constants import TaskType
from explainaboard.processors.aspect_based_sentiment_classification import (
    AspectBasedSentimentClassificationProcessor,
)
from explainaboard.processors.processor_factory import get_processor_class


class AspectBasedSentimentClassificationProcessorTest(unittest.TestCase):
    def test_get_processor_class(self) -> None:
        self.assertIs(
            get_processor_class(TaskType.aspect_based_sentiment_classification),
            AspectBasedSentimentClassificationProcessor,
        )
