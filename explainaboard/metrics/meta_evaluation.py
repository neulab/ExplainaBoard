"""Evaluation metrics for meta evaluation of natural language generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

from explainaboard.metrics.metric import (
    Metric,
    MetricConfig,
    MetricStats,
    SimpleMetricStats,
)
from explainaboard.serialization import common_registry
from explainaboard.utils.py_utils import replace_nan
from explainaboard.utils.typing_utils import narrow


@dataclass
@common_registry.register("CorrelationNLGConfig")
class CorrelationNLGConfig(MetricConfig):
    """Configuration of a correlation for general NLG tasks.

    Args:
        group_by: there are different strategies in calculating correlation for meta
                evaluation: sample level, system level and dataset level. See more
                details: https://aclanthology.org/2020.emnlp-main.751.pdf
        correlation_type: there are different types of correlation functions being used.
                So far, followings are supported: spearmanr, pearsonr, kendalltau
    """

    group_by: str = ""
    correlation_type: str = ""

    def to_metric(self) -> Metric:
        """See MetricConfig.to_metric."""
        return CorrelationNLG(self)

    def get_correlation_func(self, name: str):
        """Get correlation function based on function name.

        Args:
            name: function name
        """
        if name == "spearmanr":
            return stats.spearmanr
        elif name == "pearsonr":
            return stats.pearsonr
        elif name == "kendalltau":
            return stats.kendalltau
        else:
            raise ValueError(f"The correlation function {name} hasn't been supported")


class CorrelationNLG(Metric):
    """A metric that calculates correlations."""

    def is_simple_average(self, stats: MetricStats) -> bool:
        """See Metric.is_simple_average."""
        return False

    def uses_customized_aggregate(self) -> bool:
        """See Metric.uses_customized_aggregate."""
        return True

    def calc_stats_from_data(
        self,
        true_data: list[Any],
        pred_data: list[Any],
    ) -> MetricStats:
        """See Metric.calc_stats_from_data."""
        config = narrow(CorrelationNLGConfig, self.config)
        if config.group_by == "sample":
            corr_func = config.get_correlation_func(config.correlation_type)
            return SimpleMetricStats(
                np.array(
                    [
                        replace_nan(corr_func(true, pred)[0], 0.0)
                        for true, pred in zip(true_data, pred_data)
                    ]
                )
            )
        elif config.group_by == "system":
            return SimpleMetricStats(
                np.array([true + pred for true, pred in zip(true_data, pred_data)])
            )
        elif config.group_by == "dataset":
            return SimpleMetricStats(
                np.array(
                    [[true[0], pred[0]] for true, pred in zip(true_data, pred_data)]
                )
            )
        else:

            raise ValueError(
                f"group_by with the value {config.group_by} hasn't been supported."
            )

    def _aggregate_stats(self, stats: MetricStats) -> np.ndarray:
        """See Metric.aggregate_stats."""
        return stats.get_batch_data() if stats.is_batched() else stats.get_data()

    def stats_ndim(self) -> int:
        """See Metric.stats_ndim."""
        return 2

    def _calc_metric_from_aggregate_single(self, single_stat: np.ndarray) -> float:
        """Calculate an aggregate correlation metric from a single segment or system.

        Args:
            single_stat: The stats for the single segment or system and its dimension
                should be 2 or 3.

        Returns:
            The aggregated metric value.
        """
        val = 0.0
        config = narrow(CorrelationNLGConfig, self.config)
        corr_func = config.get_correlation_func(config.correlation_type)
        if config.group_by == "dataset":
            val = replace_nan(corr_func(single_stat[:, 0], single_stat[0:, 1])[0], 0.0)
        elif config.group_by == "sample":
            val = np.mean(single_stat)
        elif config.group_by == "system":
            n_systems = int(single_stat.shape[-1] / 2)
            true_scores = np.sum(single_stat[:, 0:n_systems], axis=0)
            pred_scores = np.sum(single_stat[:, n_systems:], axis=0)
            val = replace_nan(corr_func(true_scores, pred_scores)[0], 0.0)
        else:
            raise ValueError(
                f"group_by with the value {config.group_by} hasn't been supported."
            )
        return val

    def _calc_metric_from_aggregate(self, agg_stats: np.ndarray) -> np.ndarray:
        """See Metric.calc_metric_from_aggregate."""
        if agg_stats.ndim == self.stats_ndim():
            val = self._calc_metric_from_aggregate_single(agg_stats)
            return np.array(val)
        else:
            n_samples = agg_stats.shape[0]
            ret_metric = np.zeros(n_samples)
            for i, single_stat in enumerate(agg_stats):
                val = self._calc_metric_from_aggregate_single(single_stat)
                ret_metric[i] = val
            return ret_metric
