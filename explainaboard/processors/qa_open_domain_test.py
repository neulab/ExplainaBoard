"""Tests for explainaboard.processors.qa_open_domain"""

from __future__ import annotations

import unittest

from explainaboard import TaskType
from explainaboard.processors.processor_registry import get_processor
from explainaboard.processors.qa_open_domain import QAOpenDomainProcessor


class QAOpenDomainProcessorTest(unittest.TestCase):
    def test_get_processor(self) -> None:
        self.assertIsInstance(
            get_processor(TaskType.qa_open_domain.value), QAOpenDomainProcessor
        )
