"""Tests for explainaboard.loaders.aspect_based_sentiment_classification."""

from __future__ import annotations

import unittest

from explainaboard.constants import TaskType
from explainaboard.loaders.aspect_based_sentiment_classification import (
    AspectBasedSentimentClassificationLoader,
)
from explainaboard.loaders.loader_factory import get_loader_class


class AspectBasedSentimentClassificationTest(unittest.TestCase):
    def test_get_loader_class(self) -> None:
        self.assertIs(
            get_loader_class(TaskType.aspect_based_sentiment_classification),
            AspectBasedSentimentClassificationLoader,
        )
