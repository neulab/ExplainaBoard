"""Tests for explainaboard.processors.kg_link_tail_prediction"""

from __future__ import annotations

import unittest

from explainaboard import TaskType
from explainaboard.processors.kg_link_tail_prediction import (
    KGLinkTailPredictionProcessor,
)
from explainaboard.processors.processor_registry import get_processor


class KGLinkTailPredictionProcessorTest(unittest.TestCase):
    def test_get_processor(self) -> None:
        self.assertIsInstance(
            get_processor(TaskType.kg_link_tail_prediction.value),
            KGLinkTailPredictionProcessor,
        )
