"""Tests for explainaboard.processors.cloze_multiple_choice"""

from __future__ import annotations

import unittest

from explainaboard.constants import TaskType
from explainaboard.processors.cloze_multiple_choice import ClozeMultipleChoiceProcessor
from explainaboard.processors.processor_factory import get_processor_class


class ClozeMultipleChoiceProcessorTest(unittest.TestCase):
    def test_get_processor_class(self) -> None:
        self.assertIs(
            get_processor_class(TaskType.cloze_mutiple_choice),
            ClozeMultipleChoiceProcessor,
        )
