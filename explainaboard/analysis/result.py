"""Classes to store information about evaluation and analysis results."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Optional

from explainaboard.analysis.analyses import AnalysisResult
from explainaboard.analysis.performance import Performance


@dataclass
class Result:
    """A class to store results.

    Attributes:
        overall: Overall results. The first list is over analysis levels, and the second
          is over metrics applicable to that analysis level.
        analyses: The results of various analyses.
    """

    overall: Optional[list[list[Performance]]] = None
    analyses: Optional[list[AnalysisResult]] = None

    @classmethod
    def dict_conv(cls, k: str, v: list[list[dict]] | None):
        """A utility function for deserialization."""
        if k == 'overall':
            return [[Performance.from_dict(v2) for v2 in v1] for v1 in v] if v else None
        elif k == 'analyses':
            return [AnalysisResult.from_dict(v1) for v1 in v] if v else None
        else:
            raise NotImplementedError

    @classmethod
    def from_dict(cls, data_dict: dict) -> Result:
        """Deserialization function."""
        field_names = set(f.name for f in dataclasses.fields(cls))
        return cls(
            **{k: cls.dict_conv(k, v) for k, v in data_dict.items() if k in field_names}
        )
