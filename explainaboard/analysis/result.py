"""Classes to store information about evaluation and analysis results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import final

from explainaboard.analysis.analyses import AnalysisResult
from explainaboard.metrics.metric import MetricResult
from explainaboard.serialization import common_registry
from explainaboard.serialization.types import Serializable, SerializableData
from explainaboard.utils.typing_utils import narrow


@common_registry.register("Result")
@final
@dataclass
class Result(Serializable):
    """A class to store results.

    Attributes:
        overall: Overall results. The first dict is over analysis levels, and the second
          is over metrics applicable to that analysis level.
        analyses: The results of various analyses.
    """

    overall: dict[str, dict[str, MetricResult]]
    analyses: list[AnalysisResult]

    def serialize(self) -> dict[str, SerializableData]:
        """Implements Serializable.serialize."""
        return {
            "overall": self.overall,
            "analyses": self.analyses,
        }

    @classmethod
    def deserialize(cls, data: dict[str, SerializableData]) -> Serializable:
        """Implements Serializable.deserialize."""
        overall: dict[str, dict[str, MetricResult]] = {
            narrow(str, level_name): {
                narrow(str, metric_name): narrow(MetricResult, result)
                for metric_name, result in narrow(dict, metrics).items()
            }
            for level_name, metrics in narrow(dict, data["overall"]).items()
        }
        analyses = [narrow(AnalysisResult, x) for x in narrow(list, data["analyses"])]

        return cls(overall=overall, analyses=analyses)
