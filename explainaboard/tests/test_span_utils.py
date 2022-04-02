from __future__ import annotations

import unittest

from explainaboard.utils.span_utils import get_spans_from_bio


class TestSpanUtils(unittest.TestCase):
    def test_bio_seq_to_spans(self):
        spans = get_spans_from_bio(['O', 'B-PER', 'I-PER', 'B-ORG', 'O'])
        self.assertEqual(spans, [('PER', 1, 3), ('ORG', 3, 4)])
