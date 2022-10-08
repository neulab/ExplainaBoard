"""Tests for explainaboard.processors.machine_translation"""

from __future__ import annotations

import unittest

from explainaboard.constants import TaskType
from explainaboard.processors.machine_translation import MachineTranslationProcessor
from explainaboard.processors.processor_factory import get_processor_class


class MachineTranslationProcessorTest(unittest.TestCase):
    def test_get_processor_class(self) -> None:
        self.assertIs(
            get_processor_class(TaskType.machine_translation),
            MachineTranslationProcessor,
        )
