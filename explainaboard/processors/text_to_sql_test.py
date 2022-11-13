"""Tests for explainaboard.processors.text_to_sql"""

from __future__ import annotations

import unittest

from explainaboard.constants import TaskType
from explainaboard.processors.processor_factory import get_processor_class
from explainaboard.processors.text_to_sql import TextToSQLProcessor


class TextToSQLProcessorTest(unittest.TestCase):
    def test_get_processor_class(self) -> None:
        self.assertIs(get_processor_class(TaskType.text_to_sql), TextToSQLProcessor)
