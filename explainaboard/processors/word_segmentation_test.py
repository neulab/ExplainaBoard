"""Tests for explainaboard.processors.word_segmentation"""

from __future__ import annotations

import unittest

from explainaboard import TaskType
from explainaboard.processors.processor_registry import get_processor
from explainaboard.processors.word_segmentation import CWSProcessor


class CWSProcessorTest(unittest.TestCase):
    def test_get_processor(self) -> None:
        self.assertIsInstance(
            get_processor(TaskType.word_segmentation.value), CWSProcessor
        )
