"""Tests for explainaboard.processors.cloze_generative"""

from __future__ import annotations

import unittest

from explainaboard import TaskType
from explainaboard.processors.cloze_generative import ClozeGenerativeProcessor
from explainaboard.processors.processor_registry import get_processor


class ClozeGenerativeProcessorTest(unittest.TestCase):
    def test_get_processor(self) -> None:
        self.assertIsInstance(
            get_processor(TaskType.cloze_generative.value),
            ClozeGenerativeProcessor,
        )
