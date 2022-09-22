"""Tests for explainaboard.processors.qa_multiple_choice"""

from __future__ import annotations

import unittest

from explainaboard import TaskType
from explainaboard.processors.processor_registry import get_processor
from explainaboard.processors.qa_multiple_choice import QAMultipleChoiceProcessor


class QAMultipleChoiceProcessorTest(unittest.TestCase):
    def test_get_processor(self) -> None:
        self.assertIsInstance(
            get_processor(TaskType.qa_multiple_choice.value),
            QAMultipleChoiceProcessor,
        )
