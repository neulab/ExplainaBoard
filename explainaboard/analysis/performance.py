from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Optional

from explainaboard.analysis.case import AnalysisCase


@dataclass
class BucketPerformance:
    bucket_interval: tuple
    n_samples: float
    bucket_samples: list[Any] = field(default_factory=list)
    performances: list[Performance] = field(default_factory=list)

    @classmethod
    def dict_conv(cls, k: str, v: dict):
        if k == 'performances':
            return [Performance.from_dict(v1) for v1 in v]
        if k == 'bucket_samples' and isinstance(v, dict):
            return [AnalysisCase.from_dict(v1) for v1 in v]
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
    confidence_score_low: Optional[float] = None
    confidence_score_high: Optional[float] = None

    @classmethod
    def from_dict(cls, data_dict: dict) -> Performance:
        field_names = set(f.name for f in dataclasses.fields(cls))
        return cls(**{k: v for k, v in data_dict.items() if k in field_names})
