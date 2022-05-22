import unittest

# from dataclasses import asdict
from explainaboard.info import BucketCase, BucketCaseLabeledSpan, BucketCaseSpan


class TestBucketCaseClass(unittest.TestCase):
    def test_bucket_class_class(self):

        my_bucket_seq = BucketCase(sample_id=0)
        # print(asdict(my_bucket_seq))
        # {'sample_id': '0'}

        my_bucket_span = BucketCaseSpan(
            sample_id=0,
            token_span=(2, 3),
            char_span=(10, 25),
            text="New York",
            orig_str="input",
        )

        my_bucket_labeled_span = BucketCaseLabeledSpan(
            sample_id=0,
            token_span=(2, 3),
            char_span=(10, 25),
            text="New York",
            orig_str="input",
            true_label="LOC",
            predicted_label="ORG",
        )
        # print(asdict(my_bucket_span))
        # {'sample_id': '0', 'span': 'New York', 'true_label':
        # 'LOC', 'predicted_label': 'ORG'}

        self.assertIsNotNone(my_bucket_seq)
        self.assertIsNotNone(my_bucket_span)
        self.assertIsNotNone(my_bucket_labeled_span)
