"""Tests for explainaboard.processors.qa_tat"""

from __future__ import annotations

import unittest

from explainaboard import TaskType
from explainaboard.processors.processor_registry import get_processor
from explainaboard.processors.qa_tat import QATatProcessor


class QATatProcessorTest(unittest.TestCase):
    def test_get_processor(self) -> None:
        self.assertIsInstance(
            get_processor(TaskType.qa_tat.value),
            QATatProcessor,
        )
