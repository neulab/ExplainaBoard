"""Tests for explainaboard.analysis.analyses."""

from dataclasses import dataclass
import textwrap
from typing import final
import unittest

from explainaboard.analysis.analyses import (
    AnalysisDetails,
    AnalysisResult,
    BucketAnalysisDetails,
    CalibrationAnalysisDetails,
    ComboCountAnalysisDetails,
    ComboOccurence,
)
from explainaboard.analysis.performance import BucketPerformance
from explainaboard.metrics.metric import MetricResult, Score
from explainaboard.serialization import common_registry
from explainaboard.serialization.serializers import PrimitiveSerializer
from explainaboard.serialization.types import Serializable, SerializableData
from explainaboard.utils.typing_utils import narrow


@common_registry.register("DummyAnalysisDetails")
@final
@dataclass(frozen=True)
class DummyAnalysisDetails(AnalysisDetails):
    """Dummy AnalysisDetails implementation."""

    value: int

    def generate_report(self, name: str, level: str) -> str:
        return f"{name=}, {level=}, {self.value=}"

    def serialize(self) -> dict[str, SerializableData]:
        return {"value": self.value}

    @classmethod
    def deserialize(cls, data: dict[str, SerializableData]) -> Serializable:
        return cls(value=narrow(int, data["value"]))


class AnalysisResultTest(unittest.TestCase):
    def test_generate_report(self) -> None:
        result = AnalysisResult(
            name="foo", level="bar", details=DummyAnalysisDetails(value=42)
        )
        self.assertEqual(
            result.generate_report(), "name='foo', level='bar', self.value=42"
        )

    def test_serialization(self) -> None:
        result = AnalysisResult(
            name="foo", level="bar", details=DummyAnalysisDetails(value=42)
        )
        serialized = {
            "cls_name": "AnalysisResult",
            "name": "foo",
            "level": "bar",
            "details": {"cls_name": "DummyAnalysisDetails", "value": 42},
        }
        serializer = PrimitiveSerializer()
        self.assertEqual(serializer.serialize(result), serialized)
        self.assertEqual(serializer.deserialize(serialized), result)


class BucketAnalysisDetailsTest(unittest.TestCase):
    def test_no_performance(self) -> None:
        with self.assertRaisesRegex(ValueError, r"^No element"):
            BucketAnalysisDetails(bucket_performances=[])

    def test_inconsistent_num_metrics(self) -> None:
        perfs = [
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
        ]
        with self.assertRaisesRegex(ValueError, r"^Inconsistent metrics"):
            BucketAnalysisDetails(bucket_performances=perfs)

    def test_inconsistent_metric_names(self) -> None:
        perfs = [
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
        ]
        with self.assertRaisesRegex(ValueError, r"^Inconsistent metrics"):
            BucketAnalysisDetails(bucket_performances=perfs)

    def test_generate_report_with_interval(self) -> None:
        details = BucketAnalysisDetails(
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
        self.assertEqual(details.generate_report(name="foo", level="bar"), report)

    def test_generate_report_with_name(self) -> None:
        details = BucketAnalysisDetails(
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
        self.assertEqual(details.generate_report(name="foo", level="bar"), report)

    def test_serialization(self) -> None:
        perfs = [
            BucketPerformance(
                n_samples=5,
                bucket_samples=[0, 1, 2, 3, 4],
                results={
                    "metric": MetricResult({"score": Score(0.5)}),
                },
                bucket_name="foo",
            ),
        ]
        details = BucketAnalysisDetails(bucket_performances=perfs)

        serializer = PrimitiveSerializer()
        details_serialized = {
            "cls_name": "BucketAnalysisDetails",
            "bucket_performances": serializer.serialize(perfs),
        }
        self.assertEqual(serializer.serialize(details), details_serialized)
        self.assertEqual(serializer.deserialize(details_serialized), details)


class ComboOccurrenceTest(unittest.TestCase):
    def test_serialization(self) -> None:
        occ = ComboOccurence(("aaa", "bbb"), 3, [0, 1, 2])
        serialized = {
            "cls_name": "ComboOccurrence",
            "features": ("aaa", "bbb"),
            "sample_count": 3,
            "sample_ids": [0, 1, 2],
        }
        serializer = PrimitiveSerializer()
        self.assertEqual(serializer.serialize(occ), serialized)
        self.assertEqual(serializer.deserialize(serialized), occ)


class ComboCountAnalysisDetailsTest(unittest.TestCase):
    def test_inconsistent_feature(self) -> None:
        occs = [ComboOccurence(("xyz",), 3, list(range(3)))]
        with self.assertRaisesRegex(ValueError, r"^Inconsistent number of features"):
            ComboCountAnalysisDetails(
                features=("feat1", "feat2"),
                combo_occurrences=occs,
            )

    def test_generate_report(self) -> None:
        details = ComboCountAnalysisDetails(
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
        self.assertEqual(details.generate_report(name="foo", level="bar"), report)

    def test_serialization(self) -> None:
        occs = [
            ComboOccurence(("aaa", "bbb"), 3, list(range(3))),
        ]
        details = ComboCountAnalysisDetails(
            features=("feat1", "feat2"),
            combo_occurrences=occs,
        )
        serializer = PrimitiveSerializer()
        occs_serialized = serializer.serialize(occs)
        details_serialized = {
            "cls_name": "ComboCountAnalysisDetails",
            "features": ("feat1", "feat2"),
            "combo_occurrences": occs_serialized,
        }
        self.assertEqual(serializer.serialize(details), details_serialized)
        self.assertEqual(serializer.deserialize(details_serialized), details)


class CalibrationAnalysisDetailsTest(unittest.TestCase):
    def test_no_performance(self) -> None:
        with self.assertRaisesRegex(ValueError, r"^No element"):
            CalibrationAnalysisDetails(
                bucket_performances=[],
                expected_calibration_error=0.16,
                maximum_calibration_error=0.22,
            )

    def test_missing_accuracy_metric(self) -> None:
        perfs = [
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
        ]
        with self.assertRaisesRegex(ValueError, r"^Wrong metrics"):
            CalibrationAnalysisDetails(
                bucket_performances=perfs,
                expected_calibration_error=0.16,
                maximum_calibration_error=0.22,
            )

    def test_missing_confidence_metric(self) -> None:
        perfs = [
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
        ]
        with self.assertRaisesRegex(ValueError, r"^MetricResult does not have"):
            CalibrationAnalysisDetails(
                bucket_performances=perfs,
                expected_calibration_error=0.16,
                maximum_calibration_error=0.22,
            )

    def test_generate_report(self) -> None:
        details = CalibrationAnalysisDetails(
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
            the information of #foo#
            bucket_name\tAccuracy\t#samples
            (0.0, 0.5)\t0.5\t5
            (0.5, 1.0)\t0.7\t5

            expected_calibration_error\t0.16
            maximum_calibration_error\t0.22
            """
        )
        self.assertEqual(details.generate_report(name="foo", level="bar"), report)

    def test_serialization(self) -> None:
        perfs = [
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
        ]
        details = CalibrationAnalysisDetails(
            bucket_performances=perfs,
            expected_calibration_error=0.125,
            maximum_calibration_error=0.25,
        )
        serializer = PrimitiveSerializer()
        perfs_serialized = serializer.serialize(perfs)
        details_serialized = {
            "cls_name": "CalibrationAnalysisDetails",
            "bucket_performances": perfs_serialized,
            "expected_calibration_error": 0.125,
            "maximum_calibration_error": 0.25,
        }
        self.assertEqual(serializer.serialize(details), details_serialized)
        self.assertEqual(serializer.deserialize(details_serialized), details)
