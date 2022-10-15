"""Tests for explainaboard.processors.grammatical_error_correction"""

from __future__ import annotations

import unittest

from explainaboard.constants import TaskType
from explainaboard.processors.grammatical_error_correction import (
    GrammaticalErrorCorrectionProcessor,
)
from explainaboard.processors.processor_factory import get_processor_class


class GrammaticalErrorCorrectionProcessorTest(unittest.TestCase):
    def test_get_processor_class(self) -> None:
        self.assertIs(
            get_processor_class(TaskType.grammatical_error_correction),
            GrammaticalErrorCorrectionProcessor,
        )
