from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

from explainaboard.metrics.metric import AuxiliaryMetricResult


@dataclass
class BucketPerformance:
    bucket_interval: tuple
    n_samples: float
    bucket_samples: list[int] = field(default_factory=list)
    performances: list[Performance] = field(default_factory=list)

    @classmethod
    def dict_conv(cls, k: str, v: dict):
        """
        A deserialization utility function that takes in a key corresponding to a
        parameter name, and dictionary corresponding to a serialized version of that
        parameter's value, then return the deserialized version of the value.
        :param k: the parameter name
        :param v: the parameter's value
        """
        if k == 'performances':
            return [Performance.from_dict(v1) for v1 in v]
        else:
            return v

    @classmethod
    def from_dict(cls, data_dict: dict) -> BucketPerformance:
        field_names = set(f.name for f in dataclasses.fields(cls))
        return cls(
            **{k: cls.dict_conv(k, v) for k, v in data_dict.items() if k in field_names}
        )


@dataclass
class Performance:
    metric_name: str
    value: float
    confidence_score_low: float | None = None
    confidence_score_high: float | None = None
    auxiliary_result: AuxiliaryMetricResult | None = None

    @classmethod
    def from_dict(cls, data_dict: dict) -> Performance:
        field_names = set(f.name for f in dataclasses.fields(cls))
        return cls(**{k: v for k, v in data_dict.items() if k in field_names})
