import unittest

# from dataclasses import asdict
from explainaboard.info import BucketCaseSeq, BucketCaseSpan, BucketCaseToken


class TestBucketCaseClass(unittest.TestCase):
    def test_bucket_class_class(self):

        my_bucket_seq = BucketCaseSeq(sample_id="0")
        # print(asdict(my_bucket_seq))
        # {'sample_id': '0'}

        my_bucket_span = BucketCaseSpan(
            sample_id="0", span="New York", true_label="LOC", predicted_label="ORG"
        )
        # print(asdict(my_bucket_span))
        # {'sample_id': '0', 'span': 'New York', 'true_label':
        # 'LOC', 'predicted_label': 'ORG'}

        my_bucket_token = BucketCaseToken(sample_id="0", token_id="10")
        # print(asdict(my_bucket_token))
        # {'sample_id': '0', 'token_id': '10'}

        self.assertIsNotNone(my_bucket_seq)
        self.assertIsNotNone(my_bucket_span)
        self.assertIsNotNone(my_bucket_token)
