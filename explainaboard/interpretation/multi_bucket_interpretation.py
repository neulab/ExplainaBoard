"""Classes for Interpreting Multi-Bucket Analysis."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional

from explainaboard.analysis.analyses import BucketAnalysisResult
from explainaboard.interpretation.bucket_interpretation import BucketIntpereter
from explainaboard.interpretation.interpretation import (
    InterpretationObservation,
    InterpretationStats,
    InterpretationSuggestion,
    Interpreter,
)
from explainaboard.utils.typing_utils import narrow


@dataclass
class MultiBucketInterpretationStats(InterpretationStats):
    """Statistics of running a `MultiBucketInterpreter`.

    Attributes:
        high_correlated_features: a list of features whose values are highly correlated
         with system performances
        max_performance_gap_features: a dictionary stores the feature and performance
         gap, which is the highest across all features.
    """

    high_correlated_features: Optional[list[dict]]
    max_performance_gap_features: Optional[dict]


class MultiBucketIntpereter(Interpreter):
    """A controller to perform the interpretation process for multiple bucket analysis.

    Attributes:
        bucket_analyses: a list of bucket analyses
        feature_types: a dictionary that maps feature names to its type
         (i.e., `continuous` or `discrete`)

    """

    def __init__(
        self,
        bucket_analyses: list[BucketAnalysisResult],
        feature_types: Mapping[str, str],
    ):
        """Initialization."""
        self.bucket_analyses = [
            ba for ba in bucket_analyses if ba.cls_name == "BucketAnalysisResult"
        ]
        # This could be further optimized, i.e., remove it
        self.feature_types = feature_types
        self.threshold_related = 0.95

    def cal_stats(self) -> Mapping[str, MultiBucketInterpretationStats]:
        """Calculate Statistics."""
        high_correlated_features = {}
        max_performance_gap_features = {}
        metric_names = set()

        for analysis in self.bucket_analyses:

            ba_stats = BucketIntpereter(analysis, self.feature_types).cal_stats()
            feature_to_performance_gap = {}

            for metric_name, ba_stat in ba_stats.items():
                metric_names.add(metric_name)
                if (
                    ba_stat.correlation is not None
                    and abs(ba_stat.correlation) > self.threshold_related
                ):
                    if metric_name not in high_correlated_features:
                        high_correlated_features[metric_name] = [
                            {
                                "feature_name": analysis.name,
                                "correlation": ba_stat.correlation,
                            }
                        ]
                    else:
                        high_correlated_features[metric_name].append(
                            {
                                "feature_name": analysis.name,
                                "correlation": ba_stat.correlation,
                            }
                        )
                feature_to_performance_gap[analysis.name] = (
                    ba_stat.max_value - ba_stat.min_value
                )
            max_value_feature = max(
                feature_to_performance_gap,
                key=feature_to_performance_gap.get,  # type: ignore
            )
            max_value = feature_to_performance_gap[max_value_feature]
            max_performance_gap_features[metric_name] = {
                "feature_name": max_value_feature,
                "max_performance_gap": max_value,
            }

        multi_bucket_analysis_interpretations = {}
        for metric_name in metric_names:
            hcf = (
                high_correlated_features[metric_name]
                if metric_name in high_correlated_features
                else None
            )
            mpg = (
                max_performance_gap_features[metric_name]
                if metric_name in max_performance_gap_features
                else None
            )

            multi_bucket_analysis_interpretations[
                metric_name
            ] = MultiBucketInterpretationStats(
                high_correlated_features=hcf,
                max_performance_gap_features=mpg,
            )

        return multi_bucket_analysis_interpretations

    def default_observations_templates(
        self, stats: MultiBucketInterpretationStats
    ) -> Mapping[str, str]:
        """Definition of default observation templates."""
        templates = {}

        if stats.high_correlated_features is not None:
            template_hcf = ""
            for hcf in stats.high_correlated_features:
                template_hcf += (
                    f"the model's performance will be improved as the "
                    f"feature value of `{hcf['feature_name']}` increases. "
                    if hcf['correlation'] > 0
                    else f"the model's performance will"
                    f" be improved as the"
                    f" feature value of"
                    f" `{hcf['feature_name']}` decreases, "
                )

            template_hcf = template_hcf.rstrip(", ")
            templates["salient_feature_description"] = template_hcf

        if stats.max_performance_gap_features is not None:
            template_pgf = (
                f"On the `{stats.max_performance_gap_features['feature_name']}` "
                f"feature, the bucket performance difference reaches the maximum "
                f"of {stats.max_performance_gap_features['max_performance_gap']}"
            )

            templates["max_performance_gap_feature"] = template_pgf

        return templates

    def default_suggestions_templates(self) -> Mapping[str, str]:
        """Definition of default suggestions templates."""
        templates = {
            "salient_feature_description": "The performance of the system is "
            "highly affected by"
            " these features. Consider augment the training samples"
            " to improve the model performance.",
        }
        return templates

    def generate_observations(
        self,
        interpretation_stats: Mapping[str, InterpretationStats] | InterpretationStats,
    ) -> Mapping[str, list]:
        """Generate observations."""
        interpretation_stats = narrow(dict, interpretation_stats)
        observations_dict = {}
        for metric_name, value in interpretation_stats.items():

            templates = self.default_observations_templates(value)
            observations = [
                InterpretationObservation(keywords=keywords, content=template)
                for keywords, template in templates.items()
            ]

            observations_dict[metric_name] = observations
        return observations_dict

    def generate_suggestions(
        self,
        observations: Mapping[str, list[InterpretationObservation]]
        | list[InterpretationObservation],
    ) -> Mapping[str, list[InterpretationSuggestion]]:
        """Generate suggestions."""
        suggestions_dict = {}
        observations = narrow(dict, observations)
        for metric_name, observations_list in observations.items():
            templates = self.default_suggestions_templates()
            valid_observations = [o.keywords for o in observations_list]
            suggestions = [
                InterpretationSuggestion(keywords=keywords, content=template)
                for keywords, template in templates.items()
                if keywords in valid_observations
            ]

            suggestions_dict[metric_name] = suggestions
        return suggestions_dict
