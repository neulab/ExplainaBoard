"""Tests for explainaboard.processors.nlg_meta_evaluation"""

from __future__ import annotations

import unittest

from explainaboard import TaskType
from explainaboard.processors.nlg_meta_evaluation import NLGMetaEvaluationProcessor
from explainaboard.processors.processor_registry import get_processor


class NLGMetaEvaluationProcessorTest(unittest.TestCase):
    def test_get_processor(self) -> None:
        self.assertIsInstance(
            get_processor(TaskType.nlg_meta_evaluation.value),
            NLGMetaEvaluationProcessor,
        )
