"""Tests for explainaboard.processors.nlg_meta_evaluation"""

from __future__ import annotations

import unittest

from explainaboard.constants import TaskType
from explainaboard.processors.nlg_meta_evaluation import NLGMetaEvaluationProcessor
from explainaboard.processors.processor_factory import get_processor_class


class NLGMetaEvaluationProcessorTest(unittest.TestCase):
    def test_get_processor_class(self) -> None:
        self.assertIs(
            get_processor_class(TaskType.nlg_meta_evaluation),
            NLGMetaEvaluationProcessor,
        )
