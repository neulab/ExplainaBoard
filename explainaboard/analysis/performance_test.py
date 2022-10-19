"""Tests for explainaboard.analysis.performance."""

from __future__ import annotations

import unittest

from explainaboard.analysis.performance import BucketPerformance
from explainaboard.metrics.metric import MetricResult
from explainaboard.serialization.serializers import PrimitiveSerializer


class BucketPerformanceTest(unittest.TestCase):
    def test_post_init(self) -> None:
        # These statements should not raise any errors.
        BucketPerformance(
            n_samples=0, bucket_samples=[], results={}, bucket_interval=(1.0, 2.0)
        )
        BucketPerformance(
            n_samples=0, bucket_samples=[], results={}, bucket_name="test"
        )

    def test_post_init_invalid(self) -> None:
        with self.assertRaisesRegex(ValueError, r"^Either `bucket_interval` or"):
            BucketPerformance(n_samples=0, bucket_samples=[], results={})
        with self.assertRaisesRegex(ValueError, r"^Either `bucket_interval` or"):
            BucketPerformance(
                n_samples=0,
                bucket_samples=[],
                results={},
                bucket_interval=(1.0, 2.0),
                bucket_name="test",
            )

    def test_eq_interval(self) -> None:
        base = BucketPerformance(
            n_samples=3,
            bucket_samples=[1, 2, 3],
            results={"a": MetricResult({})},
            bucket_interval=(1.0, 2.0),
        )
        self.assertEqual(
            base,
            BucketPerformance(
                n_samples=3,
                bucket_samples=[1, 2, 3],
                results={"a": MetricResult({})},
                bucket_interval=(1.0, 2.0),
            ),
        )
        self.assertNotEqual(
            base,
            BucketPerformance(
                n_samples=5,
                bucket_samples=[1, 2, 3],
                results={"a": MetricResult({})},
                bucket_interval=(1.0, 2.0),
            ),
        )
        self.assertNotEqual(
            base,
            BucketPerformance(
                n_samples=3,
                bucket_samples=[1, 2],
                results={"a": MetricResult({})},
                bucket_interval=(1.0, 2.0),
            ),
        )
        self.assertNotEqual(
            base,
            BucketPerformance(
                n_samples=3,
                bucket_samples=[1, 2, 3],
                results={},
                bucket_interval=(1.0, 2.0),
            ),
        )
        self.assertNotEqual(
            base,
            BucketPerformance(
                n_samples=3,
                bucket_samples=[1, 2, 3],
                results={"a": MetricResult({})},
                bucket_interval=(1.0, 3.0),
            ),
        )

    def test_eq_name(self) -> None:
        base = BucketPerformance(
            n_samples=3,
            bucket_samples=[1, 2, 3],
            results={"a": MetricResult({})},
            bucket_name="test",
        )
        self.assertEqual(
            base,
            BucketPerformance(
                n_samples=3,
                bucket_samples=[1, 2, 3],
                results={"a": MetricResult({})},
                bucket_name="test",
            ),
        )
        self.assertNotEqual(
            base,
            BucketPerformance(
                n_samples=5,
                bucket_samples=[1, 2, 3],
                results={"a": MetricResult({})},
                bucket_name="test",
            ),
        )
        self.assertNotEqual(
            base,
            BucketPerformance(
                n_samples=3,
                bucket_samples=[1, 2],
                results={"a": MetricResult({})},
                bucket_name="test",
            ),
        )
        self.assertNotEqual(
            base,
            BucketPerformance(
                n_samples=3,
                bucket_samples=[1, 2, 3],
                results={},
                bucket_name="test",
            ),
        )
        self.assertNotEqual(
            base,
            BucketPerformance(
                n_samples=3,
                bucket_samples=[1, 2, 3],
                results={"a": MetricResult({})},
                bucket_name="xxx",
            ),
        )

    def test_eq_interval_name(self) -> None:
        self.assertNotEqual(
            BucketPerformance(
                n_samples=3,
                bucket_samples=[1, 2, 3],
                results={"a": MetricResult({})},
                bucket_interval=(1.0, 2.0),
            ),
            BucketPerformance(
                n_samples=3,
                bucket_samples=[1, 2, 3],
                results={"a": MetricResult({})},
                bucket_name="test",
            ),
        )

    def test_serialize_interval(self) -> None:
        result = MetricResult({})
        perf = BucketPerformance(
            n_samples=3,
            bucket_samples=[1, 2, 3],
            results={"metric": result},
            bucket_interval=(1.0, 2.0),
        )

        serializer = PrimitiveSerializer()
        serialized_result = serializer.serialize(result)
        serialized_perf = {
            "cls_name": "BucketPerformance",
            "n_samples": 3,
            "bucket_samples": [1, 2, 3],
            "results": {"metric": serialized_result},
            "bucket_interval": (1.0, 2.0),
            "bucket_name": None,
        }

        self.assertEqual(serializer.serialize(perf), serialized_perf)

    def test_serialize_name(self) -> None:
        result = MetricResult({})
        perf = BucketPerformance(
            n_samples=3,
            bucket_samples=[1, 2, 3],
            results={"metric": result},
            bucket_name="test",
        )

        serializer = PrimitiveSerializer()
        serialized_result = serializer.serialize(result)
        serialized_perf = {
            "cls_name": "BucketPerformance",
            "n_samples": 3,
            "bucket_samples": [1, 2, 3],
            "results": {"metric": serialized_result},
            "bucket_interval": None,
            "bucket_name": "test",
        }
        self.assertEqual(serializer.serialize(perf), serialized_perf)

    def test_deserialize_interval(self) -> None:
        result = MetricResult({})
        perf = BucketPerformance(
            n_samples=3,
            bucket_samples=[1, 2, 3],
            results={"metric": result},
            bucket_interval=(1.0, 2.0),
        )

        serializer = PrimitiveSerializer()
        serialized_result = serializer.serialize(result)
        serialized_perf = {
            "cls_name": "BucketPerformance",
            "n_samples": 3,
            "bucket_samples": [1, 2, 3],
            "results": {"metric": serialized_result},
            "bucket_interval": (1.0, 2.0),
            "bucket_name": None,
        }

        self.assertEqual(serializer.deserialize(serialized_perf), perf)

    def test_deserialize_name(self) -> None:
        result = MetricResult({})
        perf = BucketPerformance(
            n_samples=3,
            bucket_samples=[1, 2, 3],
            results={"metric": result},
            bucket_name="test",
        )

        serializer = PrimitiveSerializer()
        serialized_result = serializer.serialize(result)
        serialized_perf = {
            "cls_name": "BucketPerformance",
            "n_samples": 3,
            "bucket_samples": [1, 2, 3],
            "results": {"metric": serialized_result},
            "bucket_interval": None,
            "bucket_name": "test",
        }

        self.assertEqual(serializer.deserialize(serialized_perf), perf)
