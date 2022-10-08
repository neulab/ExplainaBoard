"""Tests for explainaboard.processors.argument_pair_identification"""

from __future__ import annotations

import unittest

from explainaboard.constants import TaskType
from explainaboard.processors.argument_pair_identification import (
    ArgumentPairIdentificationProcessor,
)
from explainaboard.processors.processor_factory import get_processor_class


class ArgumentPairIdentificationProcessorTest(unittest.TestCase):
    def test_get_processor_class(self) -> None:
        self.assertIs(
            get_processor_class(TaskType.argument_pair_identification),
            ArgumentPairIdentificationProcessor,
        )
