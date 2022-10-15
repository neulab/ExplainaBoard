"""Tests for explainaboard.processors.kg_link_tail_prediction"""

from __future__ import annotations

import unittest

from explainaboard.constants import TaskType
from explainaboard.processors.kg_link_tail_prediction import (
    KGLinkTailPredictionProcessor,
)
from explainaboard.processors.processor_factory import get_processor_class


class KGLinkTailPredictionProcessorTest(unittest.TestCase):
    def test_get_processor_class(self) -> None:
        self.assertIs(
            get_processor_class(TaskType.kg_link_tail_prediction),
            KGLinkTailPredictionProcessor,
        )
