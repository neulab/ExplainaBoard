"""Tests for explainaboard.processors.cloze_generative"""

from __future__ import annotations

import unittest

from explainaboard.processors.cloze_generative import ClozeGenerativeProcessor
from explainaboard.serialization.serializers import PrimitiveSerializer


class ClozeGenerativeProcessorTest(unittest.TestCase):
    def test_serialize(self) -> None:
        serializer = PrimitiveSerializer()
        self.assertEqual(
            serializer.serialize(ClozeGenerativeProcessor()),
            {"cls_name": "ClozeGenerativeProcessor"},
        )

    def test_deserialize(self) -> None:
        serializer = PrimitiveSerializer()
        self.assertIsInstance(
            serializer.deserialize({"cls_name": "ClozeGenerativeProcessor"}),
            ClozeGenerativeProcessor,
        )
