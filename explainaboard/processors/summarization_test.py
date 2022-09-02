"""Tests for explainaboard.processors.summarization"""

from __future__ import annotations

import unittest

from explainaboard import TaskType
from explainaboard.processors.processor_registry import get_processor
from explainaboard.processors.summarization import SummarizationProcessor


class SummarizationProcessorTest(unittest.TestCase):
    def test_get_processor(self) -> None:
        self.assertIsInstance(
            get_processor(TaskType.summarization.value), SummarizationProcessor
        )
