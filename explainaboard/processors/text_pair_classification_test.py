"""Tests for explainaboard.processors.text_pair_classification"""

from __future__ import annotations

import unittest

from explainaboard import TaskType
from explainaboard.processors.processor_registry import get_processor
from explainaboard.processors.text_pair_classification import (
    TextPairClassificationProcessor,
)


class TextPairClassificationProcessorTest(unittest.TestCase):
    def test_get_processor(self) -> None:
        self.assertIsInstance(
            get_processor(TaskType.text_pair_classification.value),
            TextPairClassificationProcessor,
        )
