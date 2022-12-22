"""Evaluation metrics for meta evaluation of natural language generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union

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
from explainaboard.utils.typing_utils import narrow, unwrap_or


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


@dataclass
@common_registry.register("CorrelationWMTDAConfig")
class CorrelationWMTDAConfig(MetricConfig):
    """Configuration of a correlation for WMT Metrics Meta Evaluation.

    :param group_by: Can be 'system' to group by system, 'segment' to group by segment
      or anything else (typically 'none') to not perform any grouping at all.
    :param use_z_score: Whether or not to use the z-normalized value for calculation of
      the correlation.
    :param no_human: The machine translation systems to be evaluated include automatic
      machine translation systems and human translations. The current implementation
      supports different settings with and without human translations as additional
      systems. The default setting excludes all human translations.
    """

    group_by: str = "none"
    use_z_score: bool = True
    no_human: bool = True

    def to_metric(self) -> Metric:
        """See MetricConfig.to_metric."""
        raise NotImplementedError


class CorrelationWMTDAMetric(Metric):
    """A metric that calculates correlations."""

    def is_simple_average(self, stats: MetricStats) -> bool:
        """See Metric.is_simple_average."""
        return False

    def uses_customized_aggregate(self) -> bool:
        """See Metric.uses_customized_aggregate."""
        return True

    def calc_stats_from_data(
        self,
        true_data: list[Union[str, list[str]]],
        pred_data: list[str],
    ) -> MetricStats:
        """See Metric.calc_stats_from_data."""
        config = narrow(CorrelationWMTDAConfig, self.config)
        return SimpleMetricStats(
            np.array(
                [
                    (true[0], true[1], true[3 if config.use_z_score else 2], pred)
                    for true, pred in zip(true_data, pred_data)
                ]
            )
        )

    def get_scores_from_stats(self, agg_stats: np.ndarray) -> dict[str, list]:
        """Get scores from stats.

        Args:
            agg_stats: The aggregate stats.
            config: Configuration for this metric.

        Returns:
            The score.
        """
        config = narrow(CorrelationWMTDAConfig, self.config)
        scores: dict[str, list] = {}
        for stat in agg_stats:
            sys_name = stat[0]
            seg_id = stat[1]
            manual_score = stat[2]
            auto_score = stat[3]
            group_idx = (
                sys_name
                if config.group_by == "system"
                else (seg_id if config.group_by == "segment" else "")
            )

            score = float(auto_score) if auto_score != "" else None
            score_manual = float(manual_score) if manual_score != "" else None

            if config.no_human and (
                "Human" in sys_name or "HUMAN" in sys_name or sys_name.startswith("ref")
            ):
                continue

            if score_manual is None:
                continue

            elem = (score_manual, score)
            if group_idx not in scores:
                scores[group_idx] = [elem]
            else:
                scores[group_idx].append(elem)

        return scores

    def stats_ndim(self) -> int:
        """See Metric.stats_ndim."""
        return 2

    def _aggregate_stats(self, stats: MetricStats) -> np.ndarray:
        """See Metric.aggregate_stats."""
        return stats.get_batch_data() if stats.is_batched() else stats.get_data()

    def _calc_metric_from_aggregate(self, agg_stats: np.ndarray) -> np.ndarray:
        """See Metric.calc_metric_from_aggregate."""
        if agg_stats.ndim == self.stats_ndim():
            val = self.calc_metric_from_aggregate_single(agg_stats)
            return np.array(val)
        else:
            n_samples = agg_stats.shape[0]
            ret_metric = np.zeros(n_samples)
            for i, single_stat in enumerate(agg_stats):
                val = self.calc_metric_from_aggregate_single(single_stat)
                ret_metric[i] = val
            return ret_metric

    def calc_metric_from_aggregate_single(self, single_stat: np.ndarray) -> float:
        """Calculate an aggregate correlation metric from a single segment or system.

        Args:
            single_stat: The stats for the single segment or system
            config: The configuration used in calculating the metric
        Returns:
            The aggregated metric value.
        """
        raise NotImplementedError


@dataclass
@common_registry.register("KtauCorrelationWMTDAConfig")
class KtauCorrelationWMTDAConfig(CorrelationWMTDAConfig):
    """A configuration for KtauCorrelation.

    Args:
        threshold: Following ‘Results of the WMT20 Metrics Shared Task
            (https://aclanthology.org/2020.wmt-1.77.pdf)’, to calculate segment level
            ktau
            score, we generate pairs of DA judgments attributed to distinct
            translations of the same source segment.  Distinct translations of the same
            source input whose DA scores fell within a threshold (which could have been
            deemed equal quality) were omitted from the evaluation of segment-level
            metrics. We use threshold=25 as the minimum required difference between two
            system scores to produce DARR judgments.
    """

    threshold: float = 25

    def to_metric(self) -> Metric:
        """See MetricConfig.to_metric."""
        return KtauCorrelationWMTDA(self)


class KtauCorrelationWMTDA(CorrelationWMTDAMetric):
    """A metric to calculate Kendall's Tau rank correlation."""

    def _count(self, score: list, config: Optional[MetricConfig] = None):
        config = narrow(KtauCorrelationWMTDAConfig, unwrap_or(config, self.config))
        conc = 0
        disc = 0
        num = 0
        for i in range(1, len(score)):
            for j in range(0, i):
                manual_diff = score[i][0] - score[j][0]
                system_diff = score[i][1] - score[j][1]
                if manual_diff >= config.threshold:  # i is better than system j
                    if system_diff > 0:
                        conc += 1
                    else:
                        disc += 1
                    num += 1
                elif manual_diff <= -config.threshold:  # i is worse than j
                    if system_diff < 0:
                        conc += 1
                    else:
                        disc += 1
                    num += 1
        return conc, disc, num

    def calc_metric_from_aggregate_single(self, single_stat: np.ndarray) -> float:
        """See CorrelationMetric.calc_metric_from_aggregate_single."""
        scores = self.get_scores_from_stats(single_stat)
        total_seg_num = 0
        total_conc = 0
        total_disc = 0

        for score in scores.values():
            conc, disc, num = self._count(score)
            total_seg_num += num
            total_conc += conc
            total_disc += disc

        if total_conc + total_disc != 0:
            val = (total_conc - total_disc) / (total_conc + total_disc)
        else:
            val = 0
        return val


@dataclass
@common_registry.register("PearsonCorrelationWMTDAConfig")
class PearsonCorrelationWMTDAConfig(CorrelationWMTDAConfig):
    """A configuration for the PearsonCorrelation metric."""

    def to_metric(self) -> Metric:
        """See MetricConfig.to_metric."""
        return PearsonCorrelationWMTDA(self)


class PearsonCorrelationWMTDA(CorrelationWMTDAMetric):
    """A metric to calculate Pearson's correlation."""

    def calc_metric_from_aggregate_single(self, single_stat: np.ndarray) -> float:
        """See CorrelationMetric.calc_metric_from_aggregate_single."""
        scores = self.get_scores_from_stats(single_stat)
        config = narrow(CorrelationWMTDAConfig, self.config)

        manual_score = []
        system_score = []

        if config.group_by == "segment":
            for _, group_vals in scores.items():
                for val in group_vals:
                    manual_score.append(val[0])
                    system_score.append(val[1])
        elif config.group_by == "system":
            for _, group_vals in scores.items():
                manual_scores = [val[0] for val in group_vals]
                system_scores = [val[1] for val in group_vals]
                if len(manual_scores) == 0 or len(system_scores) == 0:
                    continue
                manual_score.append(sum(manual_scores) / len(manual_scores))
                system_score.append(sum(system_scores) / len(system_scores))
        else:
            raise ValueError(
                f"The grouping way of {config.group_by} " f"hasn't been supported"
            )

        assert len(system_score) == len(manual_score)

        val = replace_nan(stats.pearsonr(system_score, manual_score)[0], 0.0)
        return val
