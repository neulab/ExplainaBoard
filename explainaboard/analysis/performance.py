"""Utility functions and classes for storing performances."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import final

from explainaboard.metrics.metric import MetricResult
from explainaboard.serialization import common_registry
from explainaboard.serialization.types import Serializable, SerializableData
from explainaboard.utils.typing_utils import narrow


@common_registry.register("BucketPerformance")
@final
@dataclass(frozen=True)
class BucketPerformance(Serializable):
    """A class containing information about performance over buckets.

    Attributes:
        n_samples: Number of samples in the bucket
        bucket_samples: IDs of the samples in the bucket
        results: A dict of MetricResults for each metric
        bucket_interval: For buckets over continuous values, the interval the bucket
          represents
        bucket_name: For buckets over discrete values, the feature the bucket represents
    """

    n_samples: int
    bucket_samples: list[int]
    results: dict[str, MetricResult]
    bucket_interval: tuple[float, float] | None = None
    bucket_name: str | None = None

    def __post_init__(self) -> None:
        """Validates member values."""
        has_interval = self.bucket_interval is not None
        has_name = self.bucket_name is not None
        if has_interval == has_name:
            raise ValueError(
                "Either `bucket_interval` or `bucket_name` must have a value, "
                "but not both. "
                f"{self.bucket_interval=}, {self.bucket_name=}"
            )

    def serialize(self) -> dict[str, SerializableData]:
        """See Serializable.serialize."""
        return {
            "n_samples": self.n_samples,
            "bucket_samples": self.bucket_samples,
            "results": self.results,
            "bucket_interval": self.bucket_interval,
            "bucket_name": self.bucket_name,
        }

    @classmethod
    def deserialize(cls, data: dict[str, SerializableData]) -> Serializable:
        """See Serializable.deserialize."""
        bucket_samples = [narrow(int, x) for x in narrow(list, data["bucket_samples"])]
        results = {
            narrow(str, k): narrow(MetricResult, v)
            for k, v in narrow(dict, data["results"]).items()
        }
        raw_bucket_interval = data.get("bucket_interval")
        if raw_bucket_interval is not None:
            assert (
                isinstance(raw_bucket_interval, Sequence)
                and len(raw_bucket_interval) == 2
            ), f"BUG: wrong bucket interval: {raw_bucket_interval=}"
            bucket_interval = (
                float(raw_bucket_interval[0]),
                float(raw_bucket_interval[1]),
            )
        else:
            bucket_interval = None
        raw_bucket_name = data.get("bucket_name")
        bucket_name = (
            narrow(str, raw_bucket_name) if raw_bucket_name is not None else None
        )

        return cls(
            n_samples=narrow(int, data["n_samples"]),
            bucket_samples=bucket_samples,
            results=results,
            bucket_interval=bucket_interval,
            bucket_name=bucket_name,
        )
