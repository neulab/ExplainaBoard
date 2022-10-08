"""Tests for explainaboard.processors.language_modeling"""

from __future__ import annotations

import unittest

from explainaboard.constants import TaskType
from explainaboard.processors.language_modeling import LanguageModelingProcessor
from explainaboard.processors.processor_factory import get_processor_class


class LanguageModelingProcessorTest(unittest.TestCase):
    def test_get_processor_class(self) -> None:
        self.assertIs(
            get_processor_class(TaskType.language_modeling), LanguageModelingProcessor
        )
