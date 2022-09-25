"""Tests for explainaboard.processors.kg_link_tail_prediction"""

from __future__ import annotations

import unittest

from explainaboard import TaskType
from explainaboard.processors.kg_link_tail_prediction import (
    KGLinkTailPredictionProcessor,
)
from explainaboard.processors.processor_registry import get_processor
from explainaboard.serialization.serializers import PrimitiveSerializer


class KGLinkTailPredictionProcessorTest(unittest.TestCase):
    def test_get_processor(self) -> None:
        self.assertIsInstance(
            get_processor(TaskType.kg_link_tail_prediction.value),
            KGLinkTailPredictionProcessor,
        )

    def test_serialize(self) -> None:
        serializer = PrimitiveSerializer()
        self.assertEqual(
            serializer.serialize(KGLinkTailPredictionProcessor()),
            {"cls_name": "KGLinkTailPredictionProcessor"},
        )

    def test_deserialize(self) -> None:
        serializer = PrimitiveSerializer()
        self.assertIsInstance(
            serializer.deserialize({"cls_name": "KGLinkTailPredictionProcessor"}),
            KGLinkTailPredictionProcessor,
        )
