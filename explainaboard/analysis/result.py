from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Optional

from explainaboard.analysis.analyses import AnalysisResult
from explainaboard.analysis.performance import Performance


@dataclass
class Result:
    overall: Optional[list[list[Performance]]] = None
    # {feature_name: {bucket_name: performance}}
    analyses: Optional[list[AnalysisResult]] = None

    @classmethod
    def dict_conv(cls, k: str, v: list[list[dict]] | None):
        if k == 'overall':
            return [[Performance.from_dict(v2) for v2 in v1] for v1 in v] if v else None
        elif k == 'analyses':
            return [AnalysisResult.from_dict(v1) for v1 in v] if v else None
        else:
            raise NotImplementedError

    @classmethod
    def from_dict(cls, data_dict: dict) -> Result:
        field_names = set(f.name for f in dataclasses.fields(cls))
        return cls(
            **{k: cls.dict_conv(k, v) for k, v in data_dict.items() if k in field_names}
        )
