from __future__ import annotations

import unittest

from explainaboard.utils.span_utils import BIOSpanOps


class TestSpanUtils(unittest.TestCase):
    def test_bio_seq_to_spans(self):

        spans = BIOSpanOps().get_spans_simple(['O', 'B-PER', 'I-PER', 'B-ORG', 'O'])
        self.assertEqual(spans, [('PER', 1, 3), ('ORG', 3, 4)])
