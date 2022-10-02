"""Tests for explainaboard.processors.meta_evaluation_wmt_da"""

from __future__ import annotations

import unittest

from explainaboard.processors.meta_evaluation_wmt_da import MetaEvaluationWMTDAProcessor
from explainaboard.serialization.serializers import PrimitiveSerializer


class NLGMetaEvaluationProcessorTest(unittest.TestCase):
    def test_serialize(self) -> None:
        serializer = PrimitiveSerializer()
        self.assertEqual(
            serializer.serialize(MetaEvaluationWMTDAProcessor()),
            {"cls_name": "MetaEvaluationWMTDAProcessor"},
        )

    def test_deserialize(self) -> None:
        serializer = PrimitiveSerializer()
        self.assertIsInstance(
            serializer.deserialize({"cls_name": "MetaEvaluationWMTDAProcessor"}),
            MetaEvaluationWMTDAProcessor,
        )
