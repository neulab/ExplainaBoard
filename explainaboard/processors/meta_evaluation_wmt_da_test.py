"""Tests for explainaboard.processors.meta_evaluation_wmt_da"""

from __future__ import annotations

import unittest

from explainaboard.constants import TaskType
from explainaboard.processors.meta_evaluation_wmt_da import MetaEvaluationWMTDAProcessor
from explainaboard.processors.processor_factory import get_processor_class


class NLGMetaEvaluationProcessorTest(unittest.TestCase):
    def test_get_processor_class(self) -> None:
        self.assertIs(
            get_processor_class(TaskType.meta_evaluation_wmt_da),
            MetaEvaluationWMTDAProcessor,
        )
