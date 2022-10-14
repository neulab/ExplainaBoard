"""Utility functions and classes for storing performances."""

from __future__ import annotations

from dataclasses import dataclass
from typing import final

from explainaboard.metrics.metric import MetricResult
from explainaboard.serialization import common_registry
from explainaboard.serialization.types import SerializableDataclass


@common_registry.register("BucketPerformance")
@final
@dataclass(frozen=True)
class BucketPerformance(SerializableDataclass):
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
