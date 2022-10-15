"""Tests for explainaboard.processors.conditional_generation"""

from __future__ import annotations

import unittest

from explainaboard.constants import TaskType
from explainaboard.processors.conditional_generation import (
    ConditionalGenerationProcessor,
)
from explainaboard.processors.processor_factory import get_processor_class


class ConditionalGenerationProcessorTest(unittest.TestCase):
    def test_get_processor_coass(self) -> None:
        self.assertIs(
            get_processor_class(TaskType.conditional_generation),
            ConditionalGenerationProcessor,
        )
