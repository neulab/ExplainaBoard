from __future__ import annotations

import unittest

from explainaboard.analysis.case import (
    AnalysisCase,
    AnalysisCaseLabeledSpan,
    AnalysisCaseSpan,
)


class AnalysisCaseClassTest(unittest.TestCase):
    def test_bucket_class_class(self):

        my_bucket_seq = AnalysisCase(sample_id=0, features={})
        # print(asdict(my_bucket_seq))
        # {'sample_id': '0'}

        my_bucket_span = AnalysisCaseSpan(
            sample_id=0,
            token_span=(2, 3),
            char_span=(10, 25),
            text="New York",
            orig_str="input",
            features={},
        )

        my_bucket_labeled_span = AnalysisCaseLabeledSpan(
            sample_id=0,
            token_span=(2, 3),
            char_span=(10, 25),
            text="New York",
            orig_str="input",
            true_label="LOC",
            predicted_label="ORG",
            features={},
        )
        # print(asdict(my_bucket_span))
        # {'sample_id': '0', 'span': 'New York', 'true_label':
        # 'LOC', 'predicted_label': 'ORG'}

        self.assertIsNotNone(my_bucket_seq)
        self.assertIsNotNone(my_bucket_span)
        self.assertIsNotNone(my_bucket_labeled_span)
