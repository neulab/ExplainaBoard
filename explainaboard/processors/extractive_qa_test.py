"""Tests for explainaboard.processors.extractive_qa"""

from __future__ import annotations

import unittest

from explainaboard import TaskType
from explainaboard.processors.extractive_qa import QAExtractiveProcessor
from explainaboard.processors.processor_registry import get_processor


class QAExtractiveProcessorTest(unittest.TestCase):
    def test_get_processor(self) -> None:
        self.assertIsInstance(
            get_processor(TaskType.qa_extractive.value), QAExtractiveProcessor
        )
