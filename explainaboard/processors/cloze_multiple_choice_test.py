"""Tests for explainaboard.processors.cloze_multiple_choice"""

from __future__ import annotations

import unittest

from explainaboard import TaskType
from explainaboard.processors.cloze_multiple_choice import ClozeMultipleChoiceProcessor
from explainaboard.processors.processor_registry import get_processor


class ClozeMultipleChoiceProcessorTest(unittest.TestCase):
    def test_get_processor(self) -> None:
        self.assertIsInstance(
            get_processor(TaskType.cloze_mutiple_choice.value),
            ClozeMultipleChoiceProcessor,
        )
