from __future__ import annotations

import unittest

from explainaboard.utils.span_utils import BIOSpanOps


class SpanUtilsTest(unittest.TestCase):
    def test_bio_spans_simple(self):

        spans = BIOSpanOps().get_spans_simple(["O", "B-PER", "I-PER", "B-ORG", "O"])
        self.assertEqual([("PER", 1, 3), ("ORG", 3, 4)], spans)

    def test_bio_spans(self):

        spans = BIOSpanOps().get_spans(
            ["O", "B-PER", "I-PER", "B-ORG", "O"],
            ["said", "Elon", "Musk", "Tesla", "CEO"],
        )
        span_tags = [x.span_tag for x in spans]
        span_poss = [x.span_pos for x in spans]
        self.assertEqual(["PER", "ORG"], span_tags)
        self.assertEqual([(1, 3), (3, 4)], span_poss)

    def test_malformed_bio_spans_simple(self):

        spans = BIOSpanOps().get_spans_simple(["I-PER", "O", "O", "I-ORG", "I-ORG"])
        self.assertEqual([("PER", 0, 1), ("ORG", 3, 5)], spans)

    def test_malformed_bio_spans(self):

        spans = BIOSpanOps().get_spans(
            ["I-PER", "O", "O", "I-ORG", "I-ORG"],
            ["Biden", "at", "the", "U.S.", "Congress"],
        )
        span_tags = [x.span_tag for x in spans]
        span_poss = [x.span_pos for x in spans]
        self.assertEqual(["PER", "ORG"], span_tags)
        self.assertEqual([(0, 1), (3, 5)], span_poss)
