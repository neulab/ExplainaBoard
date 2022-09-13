"""Utility functions and classes for storing performances."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

from explainaboard.metrics.metric import AuxiliaryMetricResult


@dataclass
class BucketPerformance:
    """A class containing information about performance over buckets.

    Attributes:
        n_samples: Number of samples in the bucket
        bucket_samples: IDs of the samples in the bucket
        performances: A list of performances for each metric
        bucket_interval: For buckets over continuous values, the interval the bucket
          represents
        bucket_name: For buckets over discrete values, the feature the bucket represents
    """

    n_samples: int
    bucket_samples: list[int] = field(default_factory=list)
    performances: list[Performance] = field(default_factory=list)
    bucket_interval: tuple[float, float] | None = None
    bucket_name: str | None = None

    @classmethod
    def dict_conv(cls, k: str, v: Any) -> Any:
        """A deserialization utility function.

        It takes in a key corresponding to a
        parameter name, and dictionary corresponding to a serialized version of that
        parameter's value, then return the deserialized version of the value.

        Args:
            k: the parameter name
            v: the parameter's value

        Returns:
            The value corresponding to the key
        """
        if k == 'performances':
            return [Performance.from_dict(v1) for v1 in v]
        else:
            return v

    @classmethod
    def from_dict(cls, data_dict: dict) -> BucketPerformance:
        """A deserialization function."""
        field_names = set(f.name for f in dataclasses.fields(cls))
        return cls(
            **{k: cls.dict_conv(k, v) for k, v in data_dict.items() if k in field_names}
        )


@dataclass
class Performance:
    """A performance value along with other information.

    Attributes:
        metric_name: The name of the metric
        value: The mean value of the metric
        confidence_score_low: The lower confidence bound
        confidence_score_high: The higher confidence bound
        auxiliary_result: Other auxiliary information used by particular metrics
    """

    metric_name: str
    value: float
    confidence_score_low: float | None = None
    confidence_score_high: float | None = None
    auxiliary_result: AuxiliaryMetricResult | None = None

    @classmethod
    def from_dict(cls, data_dict: dict) -> Performance:
        """A deserialization function."""
        field_names = set(f.name for f in dataclasses.fields(cls))
        return cls(**{k: v for k, v in data_dict.items() if k in field_names})
