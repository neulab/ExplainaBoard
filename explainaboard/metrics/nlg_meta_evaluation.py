"""Evaluation metrics for meta evaluation of natural language generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
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

    group_by: str = 'none'
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

    def stats_ndim(self) -> int:
        """See Metric.stats_dim."""
        return 2

    def _aggregate_stats(self, stats: MetricStats) -> np.ndarray:
        """See Metric.aggregate_stats."""
        return stats.get_batch_data() if stats.is_batched() else stats.get_data()

    def _calc_metric_from_aggregate(self, agg_stats: np.ndarray) -> np.ndarray:
        """See Metric.calc_metric_from_aggregate."""
        if agg_stats.ndim == 1:
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


# TODO: (1) Make Segment/System level configurable (2) Document this function
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

        val = pearsonr(system_score, manual_score)[0]

        return val
