"""Classes to store information about evaluation and analysis results."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from explainaboard.analysis.analyses import AnalysisResult
from explainaboard.metrics.metric import MetricResult
from explainaboard.serialization.serializers import PrimitiveSerializer
from explainaboard.utils.typing_utils import narrow


@dataclass
class Result:
    """A class to store results.

    Attributes:
        overall: Overall results. The first dict is over analysis levels, and the second
          is over metrics applicable to that analysis level.
        analyses: The results of various analyses.
    """

    overall: dict[str, dict[str, MetricResult]]
    analyses: list[AnalysisResult]

    @classmethod
    def dict_conv(cls, k: str, v: Any):
        """A utility function for deserialization."""
        serializer = PrimitiveSerializer()

        if k == 'overall':
            return {
                narrow(str, analysis_level_name): {
                    narrow(str, metric_name): narrow(
                        MetricResult, serializer.deserialize(metric_result)
                    )
                    for metric_name, metric_result in narrow(dict, perfs).items()
                }
                for analysis_level_name, perfs in narrow(dict, v).items()
            }
        elif k == 'analyses':
            return [AnalysisResult.from_dict(v1) for v1 in v]
        else:
            raise NotImplementedError

    @classmethod
    def from_dict(cls, data_dict: dict) -> Result:
        """Deserialization function."""
        field_names = set(f.name for f in dataclasses.fields(cls))
        return cls(
            **{k: cls.dict_conv(k, v) for k, v in data_dict.items() if k in field_names}
        )
