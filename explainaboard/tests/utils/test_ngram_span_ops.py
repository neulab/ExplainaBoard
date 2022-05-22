# import unittest
#
# from explainaboard.utils.span_utils import NgramSpanOps
#
#
# class TestSpanOps(unittest.TestCase):
#     def test_get_spans(self):
#
#         tags = ["I", "love", "New", "York"]
#
#         ngram_span_ops = NgramSpanOps(n_grams=[1, 2, 3])
#         spans, _ = ngram_span_ops.get_spans_and_match(tags=tags, tags_other=[])
#
#         span_list = [span.get_span_text for span in spans]
#         self.assertEqual(
#             span_list,
#             [
#                 'I',
#                 'love',
#                 'New',
#                 'York',
#                 'I love',
#                 'love New',
#                 'New York',
#                 'I love New',
#                 'love New York',
#             ],
#         )
