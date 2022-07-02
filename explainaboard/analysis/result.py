from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Optional

from explainaboard.analysis.analyses import AnalysisResult
from explainaboard.analysis.performance import BucketPerformance, Performance


@dataclass
class Result:
    overall: Optional[list[list[Performance]]] = None
    # {feature_name: {bucket_name: performance}}
    analyses: Optional[list[list[AnalysisResult]]] = None

    @classmethod
    def dict_conv(cls, k: str, v: dict):
        if k == 'overall':
            return {k1: Performance.from_dict(v1) for k1, v1 in v.items()}
        elif k == 'fine_grained':
            return {
                k1: [BucketPerformance.from_dict(v2) for v2 in v1]
                for k1, v1 in v.items()
            }
        elif k == 'calibration':
            return None if v is None else [Performance.from_dict(v1) for v1 in v]
        else:
            raise NotImplementedError

    @classmethod
    def from_dict(cls, data_dict: dict) -> Result:
        field_names = set(f.name for f in dataclasses.fields(cls))
        return cls(
            **{k: cls.dict_conv(k, v) for k, v in data_dict.items() if k in field_names}
        )

