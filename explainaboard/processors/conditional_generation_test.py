"""Tests for explainaboard.processors.conditional_generation"""

from __future__ import annotations

import unittest

from explainaboard import TaskType
from explainaboard.processors.conditional_generation import (
    ConditionalGenerationProcessor,
)
from explainaboard.processors.processor_registry import get_processor


class ConditionalGenerationProcessorTest(unittest.TestCase):
    def test_get_processor(self) -> None:
        self.assertIsInstance(
            get_processor(TaskType.conditional_generation.value),
            ConditionalGenerationProcessor,
        )
