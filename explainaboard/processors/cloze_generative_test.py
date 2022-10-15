"""Tests for explainaboard.processors.cloze_generative"""

from __future__ import annotations

import unittest

from explainaboard.constants import TaskType
from explainaboard.processors.cloze_generative import ClozeGenerativeProcessor
from explainaboard.processors.processor_factory import get_processor_class


class ClozeGenerativeProcessorTest(unittest.TestCase):
    def test_get_processor_class(self) -> None:
        self.assertIs(
            get_processor_class(TaskType.cloze_generative), ClozeGenerativeProcessor
        )
