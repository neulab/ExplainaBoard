import unittest

from explainaboard.utils.span_utils import BIOSpanOps


class TestBIOSpanOps(unittest.TestCase):
    def test_get_spans(self):

        tags = ["O", "O", "B-LOC", "I-LOC", "O", "B-LOC"]
        toks = ["I", "love", "New", "York", "and", "Beijing"]

        bio_span_ops = BIOSpanOps()
        spans = bio_span_ops.get_spans(tags=tags, seq=toks)

        span_text_list = [span.get_span_text for span in spans]
        span_tag_list = [span.get_span_tag for span in spans]

        # print(span_text_list)
        # print(span_tag_list)

        self.assertEqual(span_text_list, ['New York', 'Beijing'])
        self.assertEqual(span_tag_list, ['LOC', 'LOC'])

    def test_get_matched_spans(self):

        # Span a
        tags = ["O", "O", "B-LOC", "I-LOC", "O", "B-LOC"]
        toks = ["I", "love", "New", "York", "and", "Beijing"]
        bio_span_ops = BIOSpanOps()
        spans_a = bio_span_ops.get_spans(tags=tags, seq=toks)

        # Span b
        tags = ["O", "O", "B-ORG", "I-ORG", "O", "B-LOC"]
        toks = ["I", "love", "New", "York", "and", "Beijing"]
        bio_span_ops = BIOSpanOps()
        spans_b = bio_span_ops.get_spans(tags=tags, seq=toks)

        # Span c
        tags = ["O", "B-ORG", "I-ORG", "O", "B-LOC"]
        toks = ["loving", "New", "York", "and", "Beijing"]
        bio_span_ops = BIOSpanOps()
        spans_c = bio_span_ops.get_spans(tags=tags, seq=toks)

        print([span.get_span_text for span in spans_a])
        print([span.get_span_text for span in spans_b])

        print("-------1-----------")
        bio_span_ops.set_match_type("text")
        a_ind, b_ind, a_matched, b_matched = bio_span_ops.get_matched_spans(
            spans_a, spans_b
        )
        print([span.get_span_text for span in a_matched])
        self.assertEqual(
            [span.get_span_text for span in a_matched], ['New York', 'Beijing']
        )

        print("-------2----------")
        bio_span_ops.set_match_type("tag")
        a_ind, b_ind, a_matched, b_matched = bio_span_ops.get_matched_spans(
            spans_a, spans_b
        )
        print([span.get_span_text for span in a_matched])
        self.assertEqual(
            [span.get_span_text for span in a_matched], ['New York', 'Beijing']
        )

        print("-------3-----------")
        bio_span_ops.set_match_type("text_tag")
        a_ind, b_ind, a_matched, b_matched = bio_span_ops.get_matched_spans(
            spans_a, spans_b
        )
        print([span.get_span_text for span in a_matched])
        self.assertEqual([span.get_span_text for span in a_matched], ['Beijing'])

        print("-------4-----------")
        bio_span_ops.set_match_type("text_tag")
        b_ind, c_ind, b_matched, c_matched = bio_span_ops.get_matched_spans(
            spans_b, spans_c
        )
        print([span.get_span_text for span in b_matched])
        self.assertEqual(
            [span.get_span_text for span in b_matched], ['New York', 'Beijing']
        )

        print("-------5-----------")
        bio_span_ops.set_match_type("position_tag")
        b_ind, c_ind, b_matched, c_matched = bio_span_ops.get_matched_spans(
            spans_b, spans_c
        )
        print([span.get_span_text for span in b_matched])
        self.assertEqual([span.get_span_text for span in b_matched], [])
