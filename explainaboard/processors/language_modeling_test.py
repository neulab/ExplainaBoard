"""Tests for explainaboard.processors.language_modeling"""

from __future__ import annotations

import unittest

from explainaboard import TaskType
from explainaboard.processors.language_modeling import LanguageModelingProcessor
from explainaboard.processors.processor_registry import get_processor


class LanguageModelingProcessorTest(unittest.TestCase):
    def test_get_processor(self) -> None:
        self.assertIsInstance(
            get_processor(TaskType.language_modeling.value),
            LanguageModelingProcessor,
        )
