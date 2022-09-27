"""Tests for explainaboard.processors.qa_tat"""

from __future__ import annotations

import unittest

from explainaboard.processors.qa_tat import QATatProcessor
from explainaboard.serialization.serializers import PrimitiveSerializer


class QATatProcessorTest(unittest.TestCase):
    def test_serialize(self) -> None:
        serializer = PrimitiveSerializer()
        self.assertEqual(
            serializer.serialize(QATatProcessor()),
            {"cls_name": "QATatProcessor"},
        )

    def test_deserialize(self) -> None:
        serializer = PrimitiveSerializer()
        self.assertIsInstance(
            serializer.deserialize({"cls_name": "QATatProcessor"}),
            QATatProcessor,
        )
