"""Tests for explainaboard.processors.argument_pair_extraction"""

from __future__ import annotations

import unittest

from explainaboard.constants import TaskType
from explainaboard.processors.argument_pair_extraction import (
    ArgumentPairExtractionProcessor,
)
from explainaboard.processors.processor_factory import get_processor_class


class ArgumentPairExtractionProcessorTest(unittest.TestCase):
    def test_get_processor_class(self) -> None:
        self.assertIs(
            get_processor_class(TaskType.argument_pair_extraction),
            ArgumentPairExtractionProcessor,
        )
