"""Tests for explainaboard.analysis.analyses."""

import textwrap
import unittest

from explainaboard.analysis.analyses import (
    BucketAnalysisResult,
    ComboCountAnalysisResult,
)
from explainaboard.analysis.performance import BucketPerformance, Performance


class BucketAnalysisResultTest(unittest.TestCase):
    def test_inconsistent_num_metrics(self) -> None:
        with self.assertRaisesRegex(ValueError, r"^Inconsistent number of metrics"):
            BucketAnalysisResult(
                name="foo",
                level="bar",
                bucket_performances=[
                    BucketPerformance(
                        n_samples=5,
                        bucket_samples=[0, 1, 2, 3, 4],
                        performances=[
                            Performance(metric_name="metric1", value=0.5),
                            Performance(metric_name="metric2", value=0.25),
                        ],
                        bucket_name="baz",
                    ),
                    BucketPerformance(
                        n_samples=5,
                        bucket_samples=[5, 6, 7, 8, 9],
                        performances=[
                            Performance(metric_name="metric1", value=0.125),
                        ],
                        bucket_name="qux",
                    ),
                ],
            )

    def test_inconsistent_metric_names(self) -> None:
        with self.assertRaisesRegex(ValueError, r"^Inconsistent metric names"):
            BucketAnalysisResult(
                name="foo",
                level="bar",
                bucket_performances=[
                    BucketPerformance(
                        n_samples=5,
                        bucket_samples=[0, 1, 2, 3, 4],
                        performances=[
                            Performance(metric_name="metric1", value=0.5),
                            Performance(metric_name="metric2", value=0.25),
                        ],
                        bucket_name="baz",
                    ),
                    BucketPerformance(
                        n_samples=5,
                        bucket_samples=[5, 6, 7, 8, 9],
                        performances=[
                            Performance(metric_name="metric1", value=0.125),
                            Performance(metric_name="xxx", value=0.25),
                        ],
                        bucket_name="qux",
                    ),
                ],
            )

    def test_generate_report_with_interval(self) -> None:
        result = BucketAnalysisResult(
            name="foo",
            level="bar",
            bucket_performances=[
                BucketPerformance(
                    n_samples=5,
                    bucket_samples=[0, 1, 2, 3, 4],
                    performances=[
                        Performance(metric_name="metric1", value=0.5),
                        Performance(metric_name="metric2", value=0.25),
                    ],
                    bucket_interval=(1.0, 2.0),
                ),
                BucketPerformance(
                    n_samples=5,
                    bucket_samples=[5, 6, 7, 8, 9],
                    performances=[
                        Performance(metric_name="metric1", value=0.125),
                        Performance(metric_name="metric2", value=0.0625),
                    ],
                    bucket_interval=(2.0, 3.0),
                ),
            ],
        )
        report = textwrap.dedent(
            """\
            the information of #foo#
            bucket_name\tmetric1\t#samples
            (1.0, 2.0)\t0.5\t5
            (2.0, 3.0)\t0.125\t5

            the information of #foo#
            bucket_name\tmetric2\t#samples
            (1.0, 2.0)\t0.25\t5
            (2.0, 3.0)\t0.0625\t5
            """
        )
        self.assertEqual(result.generate_report(), report)

    def test_generate_report_with_name(self) -> None:
        result = BucketAnalysisResult(
            name="foo",
            level="bar",
            bucket_performances=[
                BucketPerformance(
                    n_samples=5,
                    bucket_samples=[0, 1, 2, 3, 4],
                    performances=[
                        Performance(metric_name="metric1", value=0.5),
                        Performance(metric_name="metric2", value=0.25),
                    ],
                    bucket_name="baz",
                ),
                BucketPerformance(
                    n_samples=5,
                    bucket_samples=[5, 6, 7, 8, 9],
                    performances=[
                        Performance(metric_name="metric1", value=0.125),
                        Performance(metric_name="metric2", value=0.0625),
                    ],
                    bucket_name="qux",
                ),
            ],
        )
        report = textwrap.dedent(
            """\
            the information of #foo#
            bucket_name\tmetric1\t#samples
            baz\t0.5\t5
            qux\t0.125\t5

            the information of #foo#
            bucket_name\tmetric2\t#samples
            baz\t0.25\t5
            qux\t0.0625\t5
            """
        )
        self.assertEqual(result.generate_report(), report)


class ComboCountAnalysisResultTest(unittest.TestCase):
    def test_inconsistent_feature(self) -> None:
        with self.assertRaisesRegex(ValueError, r"^Inconsistent number of features"):
            ComboCountAnalysisResult(
                name="foo",
                level="bar",
                features=("feat1", "feat2"),
                combo_counts=[(("xyz",), 123)],
            )

    def test_generate_report(self) -> None:
        result = ComboCountAnalysisResult(
            name="foo",
            level="bar",
            features=("feat1", "feat2"),
            combo_counts=[
                (("aaa", "bbb"), 123),
                (("iii", "jjj"), 456),
                (("xxx", "yyy"), 789),
            ],
        )
        report = textwrap.dedent(
            """\
            feature combos for feat1, feat2
            feat1\tfeat2\t#
            aaa\tbbb\t123
            iii\tjjj\t456
            xxx\tyyy\t789
            """
        )
        self.assertEqual(result.generate_report(), report)
