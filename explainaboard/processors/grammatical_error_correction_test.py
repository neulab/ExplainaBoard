"""Tests for explainaboard.processors.grammatical_error_correction"""

from __future__ import annotations

import unittest

from explainaboard.processors.grammatical_error_correction import (
    GrammaticalErrorCorrectionProcessor,
)
from explainaboard.serialization.serializers import PrimitiveSerializer


class GrammaticalErrorCorrectionProcessorTest(unittest.TestCase):
    def test_serialize(self) -> None:
        serializer = PrimitiveSerializer()
        self.assertEqual(
            serializer.serialize(GrammaticalErrorCorrectionProcessor()),
            {"cls_name": "GrammaticalErrorCorrectionProcessor"},
        )

    def test_deserialize(self) -> None:
        serializer = PrimitiveSerializer()
        self.assertIsInstance(
            serializer.deserialize({"cls_name": "GrammaticalErrorCorrectionProcessor"}),
            GrammaticalErrorCorrectionProcessor,
        )
