"""Tests for explainaboard.processors.processor_factory"""

from __future__ import annotations

import unittest

from explainaboard import TaskType
from explainaboard.processors.processor_factory import get_processor_class
from explainaboard.processors.text_classification import TextClassificationProcessor


# NOTE(odashi): get_processor_class should be tested in unit test files for each
# Processor subclass.
class ProcessorFactoryTest(unittest.TestCase):
    def test_get_processor_class(self) -> None:
        # This test checks if the get_processor_class function is exposed appropriately.
        self.assertEqual(
            get_processor_class(TaskType.text_classification),
            TextClassificationProcessor,
        )
