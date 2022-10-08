"""Tests for explainaboard.processors.word_segmentation"""

from __future__ import annotations

import unittest

from explainaboard.constants import TaskType
from explainaboard.processors.processor_factory import get_processor_class
from explainaboard.processors.word_segmentation import CWSProcessor


class CWSProcessorTest(unittest.TestCase):
    def test_get_processor_class(self) -> None:
        self.assertIs(get_processor_class(TaskType.word_segmentation), CWSProcessor)
