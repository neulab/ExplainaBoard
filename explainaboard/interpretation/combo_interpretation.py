"""Classes for Interpreting Combo Analysis."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import operator

from explainaboard.analysis.analyses import ComboCountAnalysisResult
from explainaboard.interpretation.interpretation import (
    Interpretation,
    InterpretationObservation,
    InterpretationStats,
    InterpretationSuggestion,
    Interpreter,
)
from explainaboard.utils.typing_utils import narrow


@dataclass
class ComboInterpretation(Interpretation):
    """A base class specifying the overall results of an combo interpretation.

    Attributes:
        observations: a dictionary that maps a metric name to a list of observations
        suggestions: a dictionary that maps a metric name to a list of suggestions
    """

    observations: list[InterpretationObservation]
    suggestions: list[InterpretationSuggestion]


@dataclass
class ComboInterpretationStats(InterpretationStats):
    """Statistics of running a `ComboInterpreter`.

    Attributes:
        frequent_error_patterns: a list of feature combinations, on which the system
         mispredict heavily. The number of list element is determined by
         `self.threshold_k`.
        frequent_error_ratio: a list of float numbers to describe the error ratio of
         patterns listed in `frequent_error_patterns`.
    """

    frequent_error_patterns: list[tuple[str, ...]]
    frequent_error_ratio: list[float]


class ComboIntpereter(Interpreter):
    """A class as a controller to perform the interpretation process for combo analysis.

    Attributes:
        combo_analysis: an analysis result with ComboCountAnalysisResult type
        threshold_k: this variable determines how many error patterns will be
         selected

    """

    def __init__(self, combo_analysis: ComboCountAnalysisResult):
        """Initialization."""
        self.combo_analysis = combo_analysis
        self.threshold_k = 3

    def cal_stats(self) -> ComboInterpretationStats:
        """Calculate Statistics."""
        sorted_combos = sorted(
            [
                x
                for x in self.combo_analysis.combo_occurrences
                if x.features[0] != x.features[1]
            ],
            key=operator.attrgetter('sample_count'),
            reverse=True,
        )
        n_incorrect_predictions = sum([x.sample_count for x in sorted_combos])
        frequent_error_patterns = []
        frequent_error_ratio = []
        for ind in range(self.threshold_k):
            frequent_error_patterns.append(sorted_combos[ind].features)
            error_ration = (
                0
                if n_incorrect_predictions == 0
                else sorted_combos[ind].sample_count * 1.0 / n_incorrect_predictions
            )
            frequent_error_ratio.append(error_ration)

        return ComboInterpretationStats(
            frequent_error_patterns=frequent_error_patterns,
            frequent_error_ratio=frequent_error_ratio,
        )

    def default_observations_templates(
        self, stats: ComboInterpretationStats
    ) -> Mapping[str, str]:
        """Definition of default observation templates."""
        template = "The system tend mispredict: "
        for (true_label, pred_label), ratio in zip(
            stats.frequent_error_patterns, stats.frequent_error_ratio
        ):
            template += (
                f"the label `{true_label}` as `{pred_label}` "
                f"(percentage of total errors: {ratio}), "
            )
        template = template.rstrip(", ")

        templates = {"misprediction_description": template}

        if len(stats.frequent_error_patterns) == 0:
            del templates["misprediction_description"]

        return templates

    def default_suggestions_templates(self) -> dict[str, str]:
        """Definition of default suggestions templates."""
        templates = {
            "misprediction_description": "These samples, which are frequently "
            "mispredicted by the model, "
            "need to be prioritized for solutions."
        }
        return templates

    def generate_observations(
        self,
        interpretation_stats: Mapping[str, InterpretationStats] | InterpretationStats,
    ) -> list:
        """Generate observations."""
        interpretation_stats = narrow(ComboInterpretationStats, interpretation_stats)
        templates = self.default_observations_templates(interpretation_stats)
        observations = [
            InterpretationObservation(keywords=keywords, content=template)
            for keywords, template in templates.items()
        ]

        return observations

    def generate_suggestions(
        self,
        observations: Mapping[str, list[InterpretationObservation]]
        | list[InterpretationObservation],
    ) -> list[InterpretationSuggestion]:
        """Generate suggestions."""
        observations = narrow(list, observations)
        templates = self.default_suggestions_templates()
        valid_observations = [o.keywords for o in observations]
        suggestions = [
            InterpretationSuggestion(keywords=keywords, content=template)
            for keywords, template in templates.items()
            if keywords in valid_observations
        ]
        return suggestions
