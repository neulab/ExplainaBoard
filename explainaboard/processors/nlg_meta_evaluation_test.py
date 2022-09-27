"""Tests for explainaboard.processors.nlg_meta_evaluation"""

from __future__ import annotations

import unittest

from explainaboard.processors.nlg_meta_evaluation import NLGMetaEvaluationProcessor
from explainaboard.serialization.serializers import PrimitiveSerializer


class NLGMetaEvaluationProcessorTest(unittest.TestCase):
    def test_serialize(self) -> None:
        serializer = PrimitiveSerializer()
        self.assertEqual(
            serializer.serialize(NLGMetaEvaluationProcessor()),
            {"cls_name": "NLGMetaEvaluationProcessor"},
        )

    def test_deserialize(self) -> None:
        serializer = PrimitiveSerializer()
        self.assertIsInstance(
            serializer.deserialize({"cls_name": "NLGMetaEvaluationProcessor"}),
            NLGMetaEvaluationProcessor,
        )
