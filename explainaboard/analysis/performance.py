"""Utility functions and classes for storing performances."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast, final

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
                "Only either `bucket_interavl` or `bucket_name` must has a value. "
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
            # HACK(odashi):
            # Deserializer could pass lists rather than tuples.
            # This casting ignores the difference between tuple and list.
            bucket_interval_seq = cast(tuple, raw_bucket_interval)
            bucket_interval = (
                narrow(float, bucket_interval_seq[0]),
                narrow(float, bucket_interval_seq[1]),
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
