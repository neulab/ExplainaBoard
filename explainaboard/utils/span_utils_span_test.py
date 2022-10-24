from __future__ import annotations

import unittest

from explainaboard.utils.span_utils import Span


class SpanClassTest(unittest.TestCase):
    def test_span_class(self):
        sample_id = 0
        seq = ["I", "love", "New", "York"]
        span_text = "New York"
        span = Span(
            span_text=span_text,
            span_tag="LOC",
            span_pos=(2, 4),
            span_capitalness="first_caps",
            span_rel_pos=4 / len(seq),
            span_chars=len(span_text),
            span_tokens=len(span_text.split(" ")),
            sample_id=sample_id,
        )

        self.assertIsNotNone(span.span_text)
        self.assertIsNone(span.span_econ)
        self.assertEqual(span.span_chars, 8)
        self.assertGreater(span.span_rel_pos, 0)
