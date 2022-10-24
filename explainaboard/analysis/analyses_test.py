"""Tests for explainaboard.analysis.analyses."""

from __future__ import annotations

from dataclasses import dataclass
import textwrap
from typing import final
import unittest

from explainaboard.analysis.analyses import (
    _subsample_analysis_cases,
    AnalysisDetails,
    AnalysisLevel,
    AnalysisResult,
    BucketAnalysis,
    BucketAnalysisDetails,
    CalibrationAnalysis,
    CalibrationAnalysisDetails,
    ComboCountAnalysis,
    ComboCountAnalysisDetails,
    ComboOccurence,
)
from explainaboard.analysis.feature import DataType, FeatureType, Value
from explainaboard.analysis.performance import BucketPerformance
from explainaboard.metrics.accuracy import AccuracyConfig
from explainaboard.metrics.metric import MetricConfig, MetricResult, Score
from explainaboard.serialization import common_registry
from explainaboard.serialization.serializers import PrimitiveSerializer
from explainaboard.serialization.types import Serializable, SerializableData
from explainaboard.utils.typing_utils import narrow


class ModuleTest(unittest.TestCase):
    def test_subsample_analysis_cases(self) -> None:
        population = [1, 2, 3]

        for _ in range(10):
            # Returns a list with unique values.
            sampled = _subsample_analysis_cases(2, population)
            self.assertEqual(len(sampled), 2)
            self.assertNotEqual(sampled[0], sampled[1])

        for _ in range(10):
            # Returns exactly the same list.
            self.assertEqual(_subsample_analysis_cases(3, population), population)
            # Larger sample limit does not cause error.
            self.assertEqual(_subsample_analysis_cases(4, population), population)


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


class BucketAnalysisTest(unittest.TestCase):
    def test_serialization(self) -> None:
        analysis = BucketAnalysis(
            description="foo",
            level="bar",
            feature="baz",
            method="qux",
            num_buckets=10,
            setting=[1, 2, 3],
            sample_limit=20,
        )
        analysis_serialized = {
            "cls_name": "BucketAnalysis",
            "description": "foo",
            "level": "bar",
            "feature": "baz",
            "method": "qux",
            "num_buckets": 10,
            "setting": [1, 2, 3],
            "sample_limit": 20,
        }
        serializer = PrimitiveSerializer()
        self.assertEqual(serializer.serialize(analysis), analysis_serialized)
        self.assertEqual(serializer.deserialize(analysis_serialized), analysis)


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


class ComboCountAnalysisTest(unittest.TestCase):
    def test_serialization(self) -> None:
        analysis = ComboCountAnalysis(
            description="foo",
            level="bar",
            features=("123", "456", "789"),
            method="baz",
            sample_limit=10,
        )
        analysis_serialized = {
            "cls_name": "ComboCountAnalysis",
            "description": "foo",
            "level": "bar",
            "features": ("123", "456", "789"),
            "method": "baz",
            "sample_limit": 10,
        }
        serializer = PrimitiveSerializer()
        self.assertEqual(serializer.serialize(analysis), analysis_serialized)
        self.assertEqual(serializer.deserialize(analysis_serialized), analysis)


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


class CalibrationAnalysisTest(unittest.TestCase):
    def test_serialization(self) -> None:
        analysis = CalibrationAnalysis(
            description="foo",
            level="bar",
            feature="baz",
            num_buckets=10,
            sample_limit=20,
        )
        analysis_serialized = {
            "cls_name": "CalibrationAnalysis",
            "description": "foo",
            "level": "bar",
            "feature": "baz",
            "num_buckets": 10,
            "sample_limit": 20,
        }
        serializer = PrimitiveSerializer()
        self.assertEqual(serializer.serialize(analysis), analysis_serialized)
        self.assertEqual(serializer.deserialize(analysis_serialized), analysis)


class AnalysisLevelTest(unittest.TestCase):
    def test_serialization(self) -> None:
        features: dict[str, FeatureType] = {"foo": Value(dtype=DataType.INT)}
        metric_configs: dict[str, MetricConfig] = {"bar": AccuracyConfig()}
        level = AnalysisLevel(
            name="test", features=features, metric_configs=metric_configs
        )
        serializer = PrimitiveSerializer()
        features_serialized = serializer.serialize(features)
        metric_configs_serialized = serializer.serialize(metric_configs)
        level_serialized = {
            "cls_name": "AnalysisLevel",
            "name": "test",
            "features": features_serialized,
            "metric_configs": metric_configs_serialized,
        }
        self.assertEqual(serializer.serialize(level), level_serialized)
        self.assertEqual(serializer.deserialize(level_serialized), level)

    def test_replace_metric_configs(self) -> None:
        level = AnalysisLevel(
            name="test", features={}, metric_configs={"foo": AccuracyConfig()}
        )
        new_level = level.replace_metric_configs({"bar": AccuracyConfig()})
        self.assertIsNot(level, new_level)
        self.assertIn("foo", level.metric_configs)
        self.assertNotIn("bar", level.metric_configs)
        self.assertNotIn("foo", new_level.metric_configs)
        self.assertIn("bar", new_level.metric_configs)
        self.assertIsNot(level.metric_configs["foo"], new_level.metric_configs["bar"])
