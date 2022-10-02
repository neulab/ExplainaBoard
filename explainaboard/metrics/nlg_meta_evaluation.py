"""Evaluation metrics for meta evaluation of natural language generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from scipy import stats
from scipy.stats import pearsonr

from explainaboard.metrics.metric import (
    Metric,
    MetricConfig,
    MetricStats,
    SimpleMetricStats,
)
from explainaboard.serialization import common_registry
from explainaboard.utils.typing_utils import narrow, unwrap_or


@dataclass
@common_registry.register("NLGCorrelationConfig")
class NLGCorrelationConfig(MetricConfig):
    """Configuration of a correlation for general NLG tasks.

    Args:
        level: there are following levels: sample level, system level and dataset level
        func_name: the method to calculate correlation (e.g., Spearman)
    """

    level: str = "sample"
    func_name: str = "spearmanr"

    def to_metric(self) -> Metric:
        """See MetricConfig.to_metric."""
        return NLGCorrelation(self)

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


class NLGCorrelation(Metric):
    """A metric that calculates correlations."""

    n_samples = 0
    n_systems = 0

    def is_simple_average(self, stats: MetricStats) -> bool:
        """See Metric.is_simple_average."""
        return False

    def uses_customized_aggregate(self) -> bool:
        """See Metric.uses_customized_aggregate."""
        return True

    def calc_stats_from_data(
        self,
        true_data: list[list[float]],
        pred_data: list[list[float]],
    ) -> MetricStats:
        """See Metric.calc_stats_from_data."""
        config = narrow(NLGCorrelationConfig, self.config)
        self.n_samples = len(true_data)
        self.n_systems = len(true_data[0])

        if config.level == "sample":
            corr_func = config.get_correlation_func(config.func_name)
            return SimpleMetricStats(
                np.array(
                    [
                        corr_func(true, pred)[0]
                        for true, pred in zip(true_data, pred_data)
                    ]
                )
            )
        elif config.level == "system":
            return SimpleMetricStats(
                np.array([true + pred for true, pred in zip(true_data, pred_data)])
            )
        else:
            return SimpleMetricStats(
                np.array(
                    [[true[0], pred[0]] for true, pred in zip(true_data, pred_data)]
                )
            )


    def _aggregate_stats(self, stats: MetricStats) -> np.ndarray:
        """See Metric.aggregate_stats."""
        if stats.is_batched():
            data = stats.get_batch_data()
            return data.reshape((data.shape[0], data.shape[-2] * data.shape[-1]))
        else:
            data = stats.get_data()
            return data.reshape((data.shape[-2] * data.shape[-1]))

    def calc_metric_from_aggregate_single(self, single_stat: np.ndarray) -> float:
        """Calculate an aggregate correlation metric from a single segment or system.

        Args:
            single_stat: The stats for the single segment or system

        Returns:
            The aggregated metric value.
        """
        val = 0
        config = narrow(NLGCorrelationConfig, self.config)
        corr_func = config.get_correlation_func(config.func_name)
        if config.level == "dataset":
            val = corr_func(single_stat[:, 0], single_stat[0:, 1])[0]
        elif config.level == "sample":
            val = np.mean(single_stat)
        elif config.level == "system":
            true_scores = np.sum(single_stat[:, 0 : int(self.n_systems)], axis=0)
            pred_scores = np.sum(single_stat[:, int(self.n_systems) :], axis=0)
            val = corr_func(true_scores, pred_scores)[0]
        return val

    def _calc_metric_from_aggregate(self, agg_stats: np.ndarray) -> np.ndarray:
        """See Metric.calc_metric_from_aggregate."""
        if agg_stats.ndim == 1:
            agg_stats = agg_stats.reshape(
                (self.n_samples, int(agg_stats.shape[0] / self.n_samples))
            )
            val = self.calc_metric_from_aggregate_single(agg_stats)
            return np.array(val)
        else:
            agg_stats = agg_stats.reshape(
                (
                    agg_stats.shape[0],
                    self.n_samples,
                    int(agg_stats.shape[1] / self.n_samples),
                )
            )

            ret_metric = np.zeros(agg_stats.shape[0])
            for i, single_stat in enumerate(agg_stats):
                val = self.calc_metric_from_aggregate_single(single_stat)
                ret_metric[i] = val
            return ret_metric


@dataclass
@common_registry.register("CorrelationConfig")
class CorrelationConfig(MetricConfig):
    """Configuration for a correlation.

    :param group_by: Can be 'system' to group by system, 'segment' to group by segment
      or anything else (typically 'none') to not perform any grouping at all.
    :param use_z_score: Whether or not to use the z-normalized value for calculation of
      the correlation.
    :param no_human: The machine translation systems to be evaluated include automatic
      machine translation systems and human translations. The current implementation
      supports different settings with and without human translations as additional
      systems. The default setting excludes all human translations.
    """

    group_by: str = 'none'
    use_z_score: bool = True
    no_human: bool = True

    def to_metric(self) -> Metric:
        """See MetricConfig.to_metric."""
        raise NotImplementedError


class CorrelationMetric(Metric):
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
        config = narrow(CorrelationConfig, self.config)

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
        config = narrow(CorrelationConfig, self.config)
        scores: dict[str, list] = {}
        for stat in agg_stats:
            sys_name = stat[0]
            seg_id = stat[1]
            manual_score = stat[2]
            auto_score = stat[3]
            group_idx = (
                sys_name
                if config.group_by == 'system'
                else (seg_id if config.group_by == 'segment' else '')
            )

            score = float(auto_score) if auto_score != '' else None
            score_manual = float(manual_score) if manual_score != '' else None

            if config.no_human and (
                'Human' in sys_name or 'HUMAN' in sys_name or sys_name.startswith('ref')
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

    def _aggregate_stats(self, stats: MetricStats) -> np.ndarray:
        """See Metric.aggregate_stats."""
        if stats.is_batched():
            data = stats.get_batch_data()
            assert data.shape[-1] == 4
            return data.reshape((data.shape[0], data.shape[-2] * data.shape[-1]))
        else:
            data = stats.get_data()
            assert data.shape[-1] == 4
            return data.reshape((data.shape[-2] * data.shape[-1]))

    def _calc_metric_from_aggregate(self, agg_stats: np.ndarray) -> np.ndarray:
        """See Metric.calc_metric_from_aggregate."""
        if agg_stats.ndim == 1:
            agg_stats = agg_stats.reshape((int(agg_stats.shape[0] / 4), 4))
            val = self.calc_metric_from_aggregate_single(agg_stats)
            return np.array(val)
        else:
            n_samples = agg_stats.shape[0]
            agg_stats = agg_stats.reshape(
                (agg_stats.shape[0], int(agg_stats.shape[1] / 4), 4)
            )
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


# TODO: (1) Make Segment/System level configurable (2) Document this function
@dataclass
@common_registry.register("KtauCorrelationConfig")
class KtauCorrelationConfig(CorrelationConfig):
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
        return KtauCorrelation(self)


class KtauCorrelation(CorrelationMetric):
    """A metric to calculate Kendall's Tau rank correlation."""

    def _count(self, score: list, config: Optional[MetricConfig] = None):
        config = narrow(KtauCorrelationConfig, unwrap_or(config, self.config))
        conc = 0
        disc = 0
        num = 0
        for i in range(1, len(score)):
            for j in range(0, i):
                if abs(score[i][0] - score[j][0]) >= config.threshold:
                    manual_better = score[i][0] > score[j][0]
                    system_better = score[i][1] > score[j][1]
                    if manual_better == system_better:
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
@common_registry.register("PearsonCorrelationConfig")
class PearsonCorrelationConfig(CorrelationConfig):
    """A configuration for the PearsonCorrelation metric."""

    def to_metric(self) -> Metric:
        """See MetricConfig.to_metric."""
        return PearsonCorrelation(self)


class PearsonCorrelation(CorrelationMetric):
    """A metric to calculate Pearson's correlation."""

    def calc_metric_from_aggregate_single(self, single_stat: np.ndarray) -> float:
        """See CorrelationMetric.calc_metric_from_aggregate_single."""
        scores = self.get_scores_from_stats(single_stat)

        manual_score = []
        system_score = []

        for group_idx, group_vals in scores.items():
            if len(group_vals[0]) != 0:
                manual_score.append(sum(group_vals[0]) / len(group_vals[0]))
            else:
                manual_score.append(0)
            if len(group_vals[1]) != 0:
                system_score.append(sum(group_vals[1]) / len(group_vals[1]))
            else:
                manual_score.append(0)
        assert len(system_score) == len(manual_score)

        val = pearsonr(system_score, manual_score)[0]

        return val
