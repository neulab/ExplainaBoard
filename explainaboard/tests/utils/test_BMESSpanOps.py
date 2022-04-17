import unittest

from explainaboard.utils.span_utils import BMESSpanOps


class TestBMESSpanOps(unittest.TestCase):
    def test_get_spans(self):

        tags = ["S", "B", "E", "B", "E"]
        toks = ["我", "喜", "欢", "纽", "约"]

        span_ops = BMESSpanOps()
        spans = span_ops.get_spans(tags=tags, seq=toks)

        span_text_list = [span.get_span_text for span in spans]
        span_tag_list = [span.get_span_tag for span in spans]

        # print(span_text_list)
        # print(span_tag_list)

        self.assertEqual(span_text_list, ['我', '喜 欢', '纽 约'])
        self.assertEqual(span_tag_list, ['S', 'BE', 'BE'])

    def test_get_matched_spans(self):

        # Span a
        tags = ["S", "B", "E", "B", "E"]
        toks = ["我", "喜", "欢", "纽", "约"]
        span_ops = BMESSpanOps()
        spans_a = span_ops.get_spans(tags=tags, seq=toks)

        # Span b
        tags = ["S", "S", "S", "B", "E"]
        toks = ["我", "喜", "欢", "纽", "约"]
        span_ops = BMESSpanOps()
        spans_b = span_ops.get_spans(tags=tags, seq=toks)

        a_ind, b_ind, a_matched, b_matched = span_ops.get_matched_spans(
            spans_a, spans_b, activate_features=["span_text"]
        )
        # print([span.get_span_text for span in a_matched])
        self.assertEqual([span.get_span_text for span in a_matched], ['我', '纽 约'])