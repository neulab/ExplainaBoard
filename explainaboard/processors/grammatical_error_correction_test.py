"""Tests for explainaboard.processors.grammatical_error_correction"""

from __future__ import annotations

import unittest

from explainaboard import TaskType
from explainaboard.processors.grammatical_error_correction import (
    GrammaticalErrorCorrectionProcessor,
)
from explainaboard.processors.processor_registry import get_processor


class GrammaticalErrorCorrectionProcessorTest(unittest.TestCase):
    def test_get_processor(self) -> None:
        self.assertIsInstance(
            get_processor(TaskType.grammatical_error_correction.value),
            GrammaticalErrorCorrectionProcessor,
        )
