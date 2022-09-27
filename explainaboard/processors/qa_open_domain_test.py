"""Tests for explainaboard.processors.qa_open_domain"""

from __future__ import annotations

import unittest

from explainaboard.processors.qa_open_domain import QAOpenDomainProcessor
from explainaboard.serialization.serializers import PrimitiveSerializer


class QAOpenDomainProcessorTest(unittest.TestCase):
    def test_serialize(self) -> None:
        serializer = PrimitiveSerializer()
        self.assertEqual(
            serializer.serialize(QAOpenDomainProcessor()),
            {"cls_name": "QAOpenDomainProcessor"},
        )

    def test_deserialize(self) -> None:
        serializer = PrimitiveSerializer()
        self.assertIsInstance(
            serializer.deserialize({"cls_name": "QAOpenDomainProcessor"}),
            QAOpenDomainProcessor,
        )
