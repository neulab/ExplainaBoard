"""Tests for explainaboard.analysis.result."""

from __future__ import annotations

import unittest

from explainaboard.analysis.analyses import AnalysisResult, BucketAnalysisDetails
from explainaboard.analysis.performance import BucketPerformance
from explainaboard.analysis.result import Result
from explainaboard.metrics.metric import MetricResult
from explainaboard.serialization.serializers import PrimitiveSerializer


class ResultTest(unittest.TestCase):
    def test_serialization(self) -> None:
        overall = {"foo": {"bar": MetricResult({})}}
        analyses = [
            AnalysisResult(
                name="baz",
                level="qux",
                details=BucketAnalysisDetails(
                    bucket_performances=[
                        BucketPerformance(
                            n_samples=5,
                            bucket_samples=[0, 1, 2, 3, 4],
                            results={},
                            bucket_interval=(0.0, 0.5),
                        )
                    ]
                ),
            )
        ]
        result = Result(overall=overall, analyses=analyses)
        serializer = PrimitiveSerializer()
        overall_serialized = serializer.serialize(overall)
        analyses_serialized = serializer.serialize(analyses)
        result_serialized = {
            "cls_name": "Result",
            "overall": overall_serialized,
            "analyses": analyses_serialized,
        }
        self.assertEqual(serializer.serialize(result), result_serialized)
        self.assertEqual(serializer.deserialize(result_serialized), result)
