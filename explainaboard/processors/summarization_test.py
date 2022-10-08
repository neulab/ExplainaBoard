"""Tests for explainaboard.processors.summarization"""

from __future__ import annotations

import unittest

from explainaboard.constants import TaskType
from explainaboard.processors.processor_factory import get_processor_class
from explainaboard.processors.summarization import SummarizationProcessor


class SummarizationProcessorTest(unittest.TestCase):
    def test_get_processor_class(self) -> None:
        self.assertIs(
            get_processor_class(TaskType.summarization), SummarizationProcessor
        )
