from __future__ import annotations

import unittest

from explainaboard.utils.span_utils import BMESSpanOps


class BMESSpanOpsTest(unittest.TestCase):
    def test_get_spans(self):

        tags = ["S", "B", "E", "B", "E"]
        toks = ["我", "喜", "欢", "纽", "约"]

        span_ops = BMESSpanOps()
        spans = span_ops.get_spans(tags=tags, toks=toks)

        span_text_list = [span.get_span_text for span in spans]
        span_tag_list = [span.get_span_tag for span in spans]

        self.assertEqual(span_text_list, ["我", "喜 欢", "纽 约"])
        self.assertEqual(span_tag_list, ["S", "BE", "BE"])

    def test_get_matched_spans(self):

        # Span a
        tags = ["S", "B", "E", "B", "E"]
        toks = ["我", "喜", "欢", "纽", "约"]
        span_ops = BMESSpanOps()
        spans_a = span_ops.get_spans(tags=tags, toks=toks)

        # Span b
        tags = ["S", "S", "S", "B", "E"]
        toks = ["我", "喜", "欢", "纽", "约"]
        span_ops = BMESSpanOps()
        spans_b = span_ops.get_spans(tags=tags, toks=toks)

        a_ind, b_ind, a_matched, b_matched = span_ops.get_matched_spans(
            spans_a, spans_b
        )
        self.assertEqual([span.get_span_text for span in a_matched], ["我", "纽 约"])
