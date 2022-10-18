"""Tests for explainaboard.analysis.analyses."""

from __future__ import annotations

import textwrap
import unittest

from explainaboard.analysis.analyses import (
    BucketAnalysisResult,
    CalibrationAnalysisResult,
    ComboCountAnalysisResult,
    ComboOccurence,
)
from explainaboard.analysis.performance import BucketPerformance
from explainaboard.metrics.metric import MetricResult, Score


class BucketAnalysisResultTest(unittest.TestCase):
    def test_inconsistent_num_metrics(self) -> None:
        with self.assertRaisesRegex(ValueError, r"^Inconsistent metrics"):
            BucketAnalysisResult(
                name="foo",
                level="bar",
                bucket_performances=[
                    BucketPerformance(
                        n_samples=5,
                        bucket_samples=[0, 1, 2, 3, 4],
                        results={
                            "metric1": MetricResult({"score": Score(0.5)}),
                            "metric2": MetricResult({"score": Score(0.25)}),
                        },
                        bucket_name="baz",
                    ),
                    BucketPerformance(
                        n_samples=5,
                        bucket_samples=[5, 6, 7, 8, 9],
                        results={
                            "metric1": MetricResult({"score": Score(0.125)}),
                        },
                        bucket_name="qux",
                    ),
                ],
            )

    def test_inconsistent_metric_names(self) -> None:
        with self.assertRaisesRegex(ValueError, r"^Inconsistent metrics"):
            BucketAnalysisResult(
                name="foo",
                level="bar",
                bucket_performances=[
                    BucketPerformance(
                        n_samples=5,
                        bucket_samples=[0, 1, 2, 3, 4],
                        results={
                            "metric1": MetricResult({"score": Score(0.5)}),
                            "metric2": MetricResult({"score": Score(0.25)}),
                        },
                        bucket_name="baz",
                    ),
                    BucketPerformance(
                        n_samples=5,
                        bucket_samples=[5, 6, 7, 8, 9],
                        results={
                            "metric1": MetricResult({"score": Score(0.125)}),
                            "xxx": MetricResult({"score": Score(0.25)}),
                        },
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
                    results={
                        "metric1": MetricResult({"score": Score(0.5)}),
                        "metric2": MetricResult({"score": Score(0.25)}),
                    },
                    bucket_interval=(1.0, 2.0),
                ),
                BucketPerformance(
                    n_samples=5,
                    bucket_samples=[5, 6, 7, 8, 9],
                    results={
                        "metric1": MetricResult({"score": Score(0.125)}),
                        "metric2": MetricResult({"score": Score(0.0625)}),
                    },
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
                    results={
                        "metric1": MetricResult({"score": Score(0.5)}),
                        "metric2": MetricResult({"score": Score(0.25)}),
                    },
                    bucket_name="baz",
                ),
                BucketPerformance(
                    n_samples=5,
                    bucket_samples=[5, 6, 7, 8, 9],
                    results={
                        "metric1": MetricResult({"score": Score(0.125)}),
                        "metric2": MetricResult({"score": Score(0.0625)}),
                    },
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
                combo_occurrences=[ComboOccurence(("xyz",), 3, list(range(3)))],
            )

    def test_generate_report(self) -> None:
        result = ComboCountAnalysisResult(
            name="foo",
            level="bar",
            features=("feat1", "feat2"),
            combo_occurrences=[
                ComboOccurence(("aaa", "bbb"), 3, list(range(3))),
                ComboOccurence(("iii", "jjj"), 3, list(range(3, 6))),
                ComboOccurence(("xxx", "yyy"), 3, list(range(6, 9))),
            ],
        )
        report = textwrap.dedent(
            """\
            feature combos for feat1, feat2
            feat1\tfeat2\t#
            aaa\tbbb\t3
            iii\tjjj\t3
            xxx\tyyy\t3
            """
        )
        self.assertEqual(result.generate_report(), report)


class CalibrationAnalysisResultTest(unittest.TestCase):
    def test_missing_accuracy_metric(self) -> None:
        with self.assertRaisesRegex(ValueError, r"^Wrong metrics"):
            CalibrationAnalysisResult(
                name="foo",
                level="example",
                bucket_performances=[
                    BucketPerformance(
                        n_samples=5,
                        bucket_samples=[0, 1, 2, 3, 4],
                        results={
                            "Accuracy": MetricResult(
                                {"score": Score(0.5), "confidence": Score(0.5)}
                            ),
                        },
                        bucket_name="baz",
                    ),
                    BucketPerformance(
                        n_samples=5,
                        bucket_samples=[5, 6, 7, 8, 9],
                        results={
                            "metric1": MetricResult({"score": Score(0.5)}),
                        },
                        bucket_name="qux",
                    ),
                ],
                expected_calibration_error=0.16,
                maximum_calibration_error=0.22,
            )

    def test_missing_confidence_metric(self) -> None:
        with self.assertRaisesRegex(ValueError, r"^MetricResult does not have"):
            CalibrationAnalysisResult(
                name="foo",
                level="example",
                bucket_performances=[
                    BucketPerformance(
                        n_samples=5,
                        bucket_samples=[0, 1, 2, 3, 4],
                        results={
                            "Accuracy": MetricResult(
                                {"score": Score(0.5), "confidence": Score(0.5)}
                            ),
                        },
                        bucket_name="baz",
                    ),
                    BucketPerformance(
                        n_samples=5,
                        bucket_samples=[5, 6, 7, 8, 9],
                        results={
                            "Accuracy": MetricResult({"score": Score(0.5)}),
                        },
                        bucket_name="qux",
                    ),
                ],
                expected_calibration_error=0.16,
                maximum_calibration_error=0.22,
            )

    def test_generate_report(self) -> None:
        result = CalibrationAnalysisResult(
            name="confidence",
            level="example",
            bucket_performances=[
                BucketPerformance(
                    n_samples=5,
                    bucket_samples=[0, 1, 2, 3, 4],
                    results={
                        "Accuracy": MetricResult(
                            {"score": Score(0.5), "confidence": Score(0.5)}
                        ),
                    },
                    bucket_interval=(0.0, 0.5),
                ),
                BucketPerformance(
                    n_samples=5,
                    bucket_samples=[5, 6, 7, 8, 9],
                    results={
                        "Accuracy": MetricResult(
                            {"score": Score(0.7), "confidence": Score(0.7)}
                        ),
                    },
                    bucket_interval=(0.5, 1.0),
                ),
            ],
            expected_calibration_error=0.16,
            maximum_calibration_error=0.22,
        )
        report = textwrap.dedent(
            """\
            the information of #confidence#
            bucket_name\tAccuracy\t#samples
            (0.0, 0.5)\t0.5\t5
            (0.5, 1.0)\t0.7\t5

            expected_calibration_error\t0.16
            maximum_calibration_error\t0.22
            """
        )
        self.assertEqual(result.generate_report(), report)
