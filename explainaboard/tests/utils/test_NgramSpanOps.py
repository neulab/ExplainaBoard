import unittest

from explainaboard.utils.span_utils import NgramSpanOps


class TestSpanOps(unittest.TestCase):
    def test_get_spans(self):

        tags = ["I", "love", "New", "York"]

        ngram_span_ops = NgramSpanOps(n_grams=[1, 2, 3])
        spans = ngram_span_ops.get_spans(tags=tags)

        span_list = [span.get_span_text for span in spans]
        # print(span_list)
        self.assertEqual(
            span_list,
            [
                'I',
                'love',
                'New',
                'York',
                'I love',
                'love New',
                'New York',
                'I love New',
                'love New York',
            ],
        )

    def test_get_matched_spans(self):

        tags_a = ["I", "love", "New", "York"]
        ngram_span_ops = NgramSpanOps(n_grams=[1, 2, 3])
        spans_a = ngram_span_ops.get_spans(tags=tags_a)
        span_list_a = [span.get_span_text for span in spans_a]
        # print(span_list_a)

        self.assertEqual(
            span_list_a,
            [
                'I',
                'love',
                'New',
                'York',
                'I love',
                'love New',
                'New York',
                'I love New',
                'love New York',
            ],
        )

        tags_b = ["I", "love", "Beijing"]
        ngram_span_ops = NgramSpanOps(n_grams=[1, 2])
        spans_b = ngram_span_ops.get_spans(tags=tags_b)
        span_list_b = [span.get_span_text for span in spans_b]

        # print(span_list_b)

        self.assertEqual(
            span_list_b,
            [
                'I',
                'love',
                'Beijing',
                'I love',
                'love Beijing',
            ],
        )

        (
            matched_a_ind,
            matched_b_ind,
            matched_spans_a,
            matched_spans_b,
        ) = ngram_span_ops.get_matched_spans(
            spans_a, spans_b, activate_features=["span_text"]
        )

        matched_span_list = [span.get_span_text for span in matched_spans_a]
        # print(matched_span_list)
        self.assertEqual(matched_span_list, ['I', 'love', 'I love'])
