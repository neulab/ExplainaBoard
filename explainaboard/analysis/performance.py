"""Utility functions and classes for storing performances."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from explainaboard.metrics.metric import MetricResult
from explainaboard.serialization.serializers import PrimitiveSerializer
from explainaboard.utils.typing_utils import narrow


@dataclass(frozen=True)
class BucketPerformance:
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
        serializer = PrimitiveSerializer()

        if k == 'performances':
            return {
                name: narrow(MetricResult, serializer.deserialize(v1))
                for name, v1 in v.items()
            }
        else:
            return v

    @classmethod
    def from_dict(cls, data_dict: dict) -> BucketPerformance:
        """A deserialization function."""
        field_names = set(f.name for f in dataclasses.fields(cls))
        return cls(
            **{k: cls.dict_conv(k, v) for k, v in data_dict.items() if k in field_names}
        )
