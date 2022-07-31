from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from scipy.stats import pearsonr

from explainaboard.metrics.metric import Metric, MetricConfig, MetricStats
from explainaboard.metrics.registry import register_metric_config
from explainaboard.utils.typing_utils import narrow, unwrap_or


@dataclass
@register_metric_config
class CorrelationConfig(MetricConfig):
    """
    Configuration for a correlation.

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


class CorrelationMetric(Metric):
    def is_simple_average(self, stats: MetricStats):
        """
        Whether the evaluation score is a simple average of the sufficient statistics.
        If so the t-test is applicable, which is much more efficient. Otherwise we do
        bootstrapping to calculate confidence interval, which is slower and potentially
        less effective.
        """
        return False

    def calc_stats_from_data(
        self,
        true_data: list[Union[str, list[str]]],
        pred_data: list[str],
        config: Optional[MetricConfig] = None,
    ) -> MetricStats:
        config = narrow(CorrelationConfig, unwrap_or(config, self.config))

        return MetricStats(
            np.array(
                [
                    (true[0], true[1], true[3 if config.use_z_score else 2], pred)
                    for true, pred in zip(true_data, pred_data)
                ]
            )
        )

    def get_scores_from_stats(
        self, agg_stats: np.ndarray, config: Optional[MetricConfig] = None
    ) -> dict[str, list]:
        config = narrow(CorrelationConfig, unwrap_or(config, self.config))
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

    def aggregate_stats(self, stats: MetricStats) -> np.ndarray:
        """
        Aggregate sufficient statistics from multiple examples into a single example
        :param stats: stats for every example
        :return: aggregated stats
        """

        data = stats.get_data()
        return data

    def calc_metric_from_aggregate(
        self, agg_stats: np.ndarray, config: Optional[MetricConfig] = None
    ) -> np.ndarray:
        if len(agg_stats.shape) == 2:
            val = self.calc_metric_from_aggregate_single(agg_stats, config)
            return np.array([val])
        else:
            n_samples = agg_stats.shape[0]
            ret_metric = np.zeros(n_samples)
            for i, single_stat in enumerate(agg_stats):
                val = self.calc_metric_from_aggregate_single(single_stat, config)
                ret_metric[i] = val
            return ret_metric

    def calc_metric_from_aggregate_single(
        self, single_stat: np.ndarray, config: Optional[MetricConfig] = None
    ) -> float:
        raise NotImplementedError


# TODO: (1) Make Segment/System level configurable (2) Document this function
@dataclass
@register_metric_config
class KtauCorrelationConfig(CorrelationConfig):
    """
    :param threshold: Following ‘Results of the WMT20 Metrics Shared Task
        (https://aclanthology.org/2020.wmt-1.77.pdf)’, to calculate segment level ktau
         score, we generate pairs of DA judgments attributed to distinct
        translations of the same source segment.  Distinct translations of the same
        source input whose DA scores fell within a threshold (which could have been
        deemed equal quality) were omitted from the evaluation of segment-level
        metrics. We use threshold=25 as the minimum required difference between two
        system scores to produce DARR judgments.
    """

    threshold: float = 25

    def to_metric(self):
        return KtauCorrelation(self)


class KtauCorrelation(CorrelationMetric):
    def count(self, score: list, config: Optional[MetricConfig] = None):
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

    def calc_metric_from_aggregate_single(
        self, single_stat: np.ndarray, config: Optional[MetricConfig] = None
    ) -> float:
        scores = self.get_scores_from_stats(single_stat, config)
        total_seg_num = 0
        total_conc = 0
        total_disc = 0

        for score in scores.values():
            conc, disc, num = self.count(score, config)
            total_seg_num += num
            total_conc += conc
            total_disc += disc

        if total_conc + total_disc != 0:
            val = (total_conc - total_disc) / (total_conc + total_disc)
        else:
            val = 0
        return val


@dataclass
@register_metric_config
class PearsonCorrelationConfig(CorrelationConfig):
    def to_metric(self):
        return PearsonCorrelation(self)


class PearsonCorrelation(CorrelationMetric):
    def calc_metric_from_aggregate_single(
        self, single_stat: np.ndarray, config: Optional[MetricConfig] = None
    ) -> float:
        config = narrow(PearsonCorrelationConfig, unwrap_or(config, self.config))
        scores = self.get_scores_from_stats(single_stat, config)

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
