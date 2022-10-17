"""Classes for Interpreting Bucket Analysis."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from scipy.stats import spearmanr

from explainaboard.analysis.analyses import BucketAnalysisResult
from explainaboard.interpretation.interpretation import (
    Interpretation,
    InterpretationObservation,
    InterpretationStats,
    InterpretationSuggestion,
    Interpreter,
)
from explainaboard.utils.typing_utils import narrow


@dataclass
class BucketInterpretation(Interpretation):
    """A base class specifying the overall results of a bucket interpretation.

    Attributes:
        observations: a dictionary that maps a metric name to a list of observations
        suggestions: a dictionary that maps a metric name to a list of suggestions
    """

    observations: Mapping[str, list[InterpretationObservation]]
    suggestions: Mapping[str, list[InterpretationSuggestion]]


@dataclass
class BucketInterpretationStats(InterpretationStats):
    """Statistics of running a `BucketInterpreter`.

    Attributes:
        correlation: a float score that describe how correlated
        the system performances (e.g., F1) are with feature values (e.g., text length)
        max_value: the maximum performance of all buckets (e.g., [1,5], [5,15],
        [15, 20]) along a given feature (e.g., text length)
        min_value: the minimum performance of all buckets (e.g., [1,5], [5,15],
        [15, 20]) along a given feature (e.g., text length)
        unreliable_buckets: the buckets whose test samples are less than a threshold,
        as defined below.
    """

    correlation: float | None
    max_value: float
    min_value: float
    unreliable_buckets: list[str]


class BucketIntpereter(Interpreter):
    """A controller to perform the interpretation process for bucket analysis.

        Attributes:
            bucket_analysis: an analysis result with BucketAnalysisResult type
            feature_types: a dictionary that maps feature name to corresponding type.
            threshold_unreliable_bucket: this variable determines how many
            unreliable buckets will be selected

    """

    def __init__(
        self, bucket_analysis: BucketAnalysisResult, feature_types: Mapping[str, str]
    ):
        """Initialization."""
        self.bucket_analysis = bucket_analysis
        # This could be further optimized, i.e., remove it
        self.feature_types = feature_types
        self.threshold_unreliable_bucket = 100

    def _get_performances_d2l(self) -> Mapping[str, list]:
        """Get performances with the format of dict_to_list."""
        performances_d2l = {}
        for bucket_info in self.bucket_analysis.bucket_performances:
            for metric_name, perf_info in bucket_info.performances.items():
                if metric_name not in performances_d2l:
                    performances_d2l[metric_name] = [perf_info.value]
                else:
                    performances_d2l[metric_name].append(perf_info.value)
        return performances_d2l

    def _get_unreliable_buckets(self) -> list[str]:
        """Get unreliable buckets."""
        # This function could be merged with _get_performances_d2l at the cost of
        # understandability
        unreliable_buckets = []
        for bucket_info in self.bucket_analysis.bucket_performances:

            if bucket_info.n_samples < self.threshold_unreliable_bucket:
                if (
                    self.feature_types[self.bucket_analysis.name] == "continuous"
                    and bucket_info.bucket_interval is not None
                ):
                    bucket_name = (
                        f"({bucket_info.bucket_interval[0]}, "
                        f"{bucket_info.bucket_interval[1]})"
                    )
                else:
                    bucket_name = f"{bucket_info.bucket_name}"
                unreliable_buckets.append(bucket_name)
        return unreliable_buckets

    def cal_stats(self) -> Mapping[str, BucketInterpretationStats]:
        """Calculate Statistics."""
        performances_d2l = self._get_performances_d2l()
        unreliable_buckets = self._get_unreliable_buckets()
        bucket_analysis_stats = {}
        for metric_name, values in performances_d2l.items():
            corr = (
                spearmanr(range(len(values)), values)[0]
                if self.feature_types[self.bucket_analysis.name] == "continuous"
                else None
            )
            max_value = max(values)
            min_value = min(values)

            stats = BucketInterpretationStats(
                correlation=corr,
                max_value=max_value,
                min_value=min_value,
                unreliable_buckets=unreliable_buckets,
            )

            bucket_analysis_stats[metric_name] = stats

        return bucket_analysis_stats

    def default_observations_templates(
        self, stats: BucketInterpretationStats
    ) -> Mapping[str, str]:
        """Definition of default observation templates."""
        templates = {
            "performance_description": f"The largest performance gap between different"
            f" buckets is {stats.max_value - stats.min_value}, and the best "
            f"performance is {stats.max_value} worse "
            f"performance is {stats.min_value}",
            "correlation_description": f"The correlation between the model performance"
            f" and feature value of {self.bucket_analysis.name} is:"
            f" {stats.correlation}",
            "unreliable_buckets": f"The number of samples in these buckets"
            f"{stats.unreliable_buckets} are relatively fewer (<= 100),"
            f"which may result in a large variance.",
        }

        if stats.correlation is None:
            del templates["correlation_description"]

        if len(stats.unreliable_buckets) == 0:
            del templates["unreliable_buckets"]

        return templates

    def default_suggestions_templates(self) -> Mapping[str, str]:
        """Definition of default suggestions templates."""
        templates = {
            "correlation_description": f"If the absolute value of correlation is "
            f"greater than 0.9, "
            f"it means that the performance of the system is highly affected by"
            f" features. Consider improving the training samples under appropriate"
            f" feature value of {self.bucket_analysis.name} to improve the"
            f" model performance.",
            "unreliable_buckets": "If the performance on these unreliable are also low,"
            " please check whether the corresponding samples"
            " in the training set are fewer as well, and consider"
            " introducing more samples to further improve the performance.",
        }
        return templates

    def generate_observations(
        self,
        interpretation_stats: Mapping[str, InterpretationStats] | InterpretationStats,
    ) -> dict[str, list]:
        """Generate observations."""
        observations_dict = {}
        interpretation_stats = narrow(dict, interpretation_stats)
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
        observations = narrow(dict, observations)
        suggestions_dict = {}
        for metric_name, observation_list in observations.items():
            templates = self.default_suggestions_templates()
            valid_observations = [o.keywords for o in observation_list]
            suggestions = [
                InterpretationSuggestion(keywords=keywords, content=template)
                for keywords, template in templates.items()
                if keywords in valid_observations
            ]

            suggestions_dict[metric_name] = suggestions
        return suggestions_dict
