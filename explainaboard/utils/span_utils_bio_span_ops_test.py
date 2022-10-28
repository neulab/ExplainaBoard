from __future__ import annotations

import unittest

from explainaboard.utils.span_utils import BIOSpanOps


class BIOSpanOpsTest(unittest.TestCase):
    def test_get_spans(self):

        tags = ["O", "O", "B-LOC", "I-LOC", "O", "B-LOC"]
        toks = ["I", "love", "New", "York", "and", "Beijing"]

        bio_span_ops = BIOSpanOps()
        spans = bio_span_ops.get_spans(tags=tags, toks=toks)

        span_text_list = [span.get_span_text for span in spans]
        span_tag_list = [span.get_span_tag for span in spans]

        self.assertEqual(span_text_list, ["New York", "Beijing"])
        self.assertEqual(span_tag_list, ["LOC", "LOC"])

    def test_get_matched_spans(self):

        # Span a
        tags = ["O", "O", "B-LOC", "I-LOC", "O", "B-LOC"]
        toks = ["I", "love", "New", "York", "and", "Beijing"]
        bio_span_ops = BIOSpanOps()
        spans_a = bio_span_ops.get_spans(tags=tags, toks=toks)

        # Span b
        tags = ["O", "O", "B-ORG", "I-ORG", "O", "B-LOC"]
        toks = ["I", "love", "New", "York", "and", "Beijing"]
        bio_span_ops = BIOSpanOps()
        spans_b = bio_span_ops.get_spans(tags=tags, toks=toks)

        # Span c
        tags = ["O", "B-ORG", "I-ORG", "O", "B-LOC"]
        toks = ["loving", "New", "York", "and", "Beijing"]
        bio_span_ops = BIOSpanOps()
        spans_c = bio_span_ops.get_spans(tags=tags, toks=toks)

        bio_span_ops.set_match_type("text")
        a_ind, b_ind, a_matched, b_matched = bio_span_ops.get_matched_spans(
            spans_a, spans_b
        )
        self.assertEqual(
            [span.get_span_text for span in a_matched], ["New York", "Beijing"]
        )

        bio_span_ops.set_match_type("tag")
        a_ind, b_ind, a_matched, b_matched = bio_span_ops.get_matched_spans(
            spans_a, spans_b
        )
        self.assertEqual(
            [span.get_span_text for span in a_matched], ["New York", "Beijing"]
        )

        bio_span_ops.set_match_type("text_tag")
        a_ind, b_ind, a_matched, b_matched = bio_span_ops.get_matched_spans(
            spans_a, spans_b
        )
        self.assertEqual([span.get_span_text for span in a_matched], ["Beijing"])

        bio_span_ops.set_match_type("text_tag")
        b_ind, c_ind, b_matched, c_matched = bio_span_ops.get_matched_spans(
            spans_b, spans_c
        )
        self.assertEqual(
            [span.get_span_text for span in b_matched], ["New York", "Beijing"]
        )

        bio_span_ops.set_match_type("position_tag")
        b_ind, c_ind, b_matched, c_matched = bio_span_ops.get_matched_spans(
            spans_b, spans_c
        )
        self.assertEqual([span.get_span_text for span in b_matched], [])
