"""Tests for explainaboard.processors.machine_translation"""

from __future__ import annotations

import unittest

from explainaboard import TaskType
from explainaboard.processors.machine_translation import MachineTranslationProcessor
from explainaboard.processors.processor_registry import get_processor


class MachineTranslationProcessorTest(unittest.TestCase):
    def test_get_processor(self) -> None:
        self.assertIsInstance(
            get_processor(TaskType.machine_translation.value),
            MachineTranslationProcessor,
        )
