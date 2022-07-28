from __future__ import annotations

from dataclasses import dataclass
from typing import cast, Optional, Union

import numpy as np
from scipy.stats import pearsonr

from explainaboard.metrics.metric import Metric, MetricConfig, MetricStats
from explainaboard.metrics.registry import register_metric_config


# TODO: (1) Make Segment/System level configurable (2) Document this function
@dataclass
@register_metric_config
class SegKtauCorrConfig(MetricConfig):
    """
    no_human: The machine translation systems to be evaluated include automatic
         machine translation systems and human translations. The current implementation
        supports different settings with and without human translations as additional
        systems. The default setting excludes all human translations.
    threshold: Following ‘Results of the WMT20 Metrics Shared Task
        (https://aclanthology.org/2020.wmt-1.77.pdf)’, to calculate segment level ktau
         score, we generate pairs of DA judgments attributed to distinct
        translations of the same source segment.  Distinct translations of the same
        source input whose DA scores fell within a threshold (which could have been
        deemed equal quality) were omitted from the evaluation of segment-level
        metrics. We use threshold=25 as the minimum required difference between two
        system scores to produce DARR judgments.
    """

    no_human: bool = True
    threshold: float = 25

    def to_metric(self):
        return SegKtauCorrScore(self)


class SegKtauCorrScore(Metric):
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

        return MetricStats(
            np.array(
                [
                    (true[0], true[1], true[2], pred)
                    for true, pred in zip(true_data, pred_data)
                ]
            )
        )

    def aggregate_stats(self, stats: MetricStats) -> np.ndarray:
        """
        Aggregate sufficient statistics from multiple examples into a single example
        :param stats: stats for every example
        :return: aggregated stats
        """

        data = stats.get_data()
        return data

    def get_scores_from_stats(
        self, agg_stats: np.ndarray, config: Optional[MetricConfig] = None
    ) -> dict[str, list]:
        config = cast(SegKtauCorrConfig, self._get_config(config))
        scores: dict[str, list] = {}
        for stat in agg_stats:
            sys_name = stat[0]
            seg_id = stat[1]
            manual_score = stat[2]
            auto_score = stat[3]

            score = float(auto_score) if auto_score != '' else None
            score_manual = float(manual_score) if manual_score != '' else None

            if config.no_human and (
                'Human' in sys_name or 'HUMAN' in sys_name or sys_name.startswith('ref')
            ):
                continue

            if score_manual is None:
                continue

            if seg_id not in scores:
                scores[seg_id] = []
            scores[seg_id].append([score_manual, score])

        return scores

    def count(self, score: list, config: Optional[MetricConfig] = None):
        config = cast(SegKtauCorrConfig, self._get_config(config))
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

    def calc_metric_from_aggregate(
        self, agg_stats: np.ndarray, config: Optional[MetricConfig] = None
    ) -> float:
        if len(agg_stats.shape) == 2:
            val = self.calc_metric_from_aggregate_single(agg_stats, config)
            return val
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
class RootMeanSquareErrorConfig(MetricConfig):
    no_human: bool = True

    def to_metric(self):
        return RootMeanSquareErrorScore(self)


class RootMeanSquareErrorScore(Metric):
    def calc_stats_from_data(
        self,
        true_data: list[Union[str, list[str]]],
        pred_data: list[str],
        config: Optional[MetricConfig] = None,
    ) -> MetricStats:

        return MetricStats(
            np.array(
                [(true[0], true[3], pred) for true, pred in zip(true_data, pred_data)]
            )
        )

    def is_simple_average(self, stats: MetricStats):
        """
        Whether the evaluation score is a simple average of the sufficient statistics.
        If so the t-test is applicable, which is much more efficient. Otherwise we do
        bootstrapping to calculate confidence interval, which is slower and potentially
        less effective.
        """
        return False

    def aggregate_stats(self, stats: MetricStats) -> np.ndarray:
        """
        Aggregate sufficient statistics from multiple examples into a single example
        :param stats: stats for every example
        :return: aggregated stats
        """

        data = stats.get_data()
        return data

    def get_scores_from_stats(
        self, agg_stats: np.ndarray, config: Optional[MetricConfig] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        config = cast(RootMeanSquareErrorConfig, self._get_config(config))
        score_manuals = []
        scores = []
        for stat in agg_stats:
            sys_name = stat[0]
            manual_score = stat[1]
            auto_score = stat[2]

            score = float(auto_score) if auto_score != '' else None
            score_manual = float(manual_score) if manual_score != '' else None

            if config.no_human and (
                'Human' in sys_name or 'HUMAN' in sys_name or sys_name.startswith('ref')
            ):
                continue

            if score_manual is None:
                continue

            score_manuals.append(score_manual)
            scores.append(score)

        return np.array(score_manuals), np.array(scores)

    def normalize(self, data: np.adarray) -> np.adarray:
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def calc_metric_from_aggregate(
        self, agg_stats: np.ndarray, config: Optional[MetricConfig] = None
    ) -> float:
        if len(agg_stats.shape) == 2:
            val = self.calc_metric_from_aggregate_single(agg_stats, config)
            return val
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
        score_manuals, scores = self.get_scores_from_stats(single_stat, config)
        score_manuals = self.normalize(score_manuals)
        scores = self.normalize(scores)
        val = np.sqrt((np.square(score_manuals - scores)).mean())
        return val


@dataclass
@register_metric_config
class SysPearsonCorrConfig(MetricConfig):
    no_human: bool = True

    def to_metric(self):
        return SysPearsonCorrScore(self)


class SysPearsonCorrScore(Metric):
    def calc_stats_from_data(
        self,
        true_data: list[Union[str, list[str]]],
        pred_data: list[str],
        config: Optional[MetricConfig] = None,
    ) -> MetricStats:

        return MetricStats(
            np.array(
                [(true[0], true[3], pred) for true, pred in zip(true_data, pred_data)]
            )
        )

    def is_simple_average(self, stats: MetricStats):
        """
        Whether the evaluation score is a simple average of the sufficient statistics.
        If so the t-test is applicable, which is much more efficient. Otherwise we do
        bootstrapping to calculate confidence interval, which is slower and potentially
        less effective.
        """
        return False

    def aggregate_stats(self, stats: MetricStats) -> np.ndarray:
        """
        Aggregate sufficient statistics from multiple examples into a single example
        :param stats: stats for every example
        :return: aggregated stats
        """

        data = stats.get_data()
        return data

    def get_scores_from_stats(
        self, agg_stats: np.ndarray, config: Optional[MetricConfig] = None
    ) -> dict[str, list]:
        scores: dict[str, list] = {}
        for stat in agg_stats:
            sys_name = stat[0]
            manual_score = stat[1]
            auto_score = stat[2]

            score = float(auto_score) if auto_score != '' else None
            score_manual = float(manual_score) if manual_score != '' else None

            if sys_name not in scores:
                scores[sys_name] = [[], []]
            if score_manual is not None:
                scores[sys_name][0].append(score_manual)
            if score is not None:
                scores[sys_name][1].append(score)

        return scores

    def calc_metric_from_aggregate(
        self, agg_stats: np.ndarray, config: Optional[MetricConfig] = None
    ) -> float:
        if len(agg_stats.shape) == 2:
            val = self.calc_metric_from_aggregate_single(agg_stats, config)
            return val
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
        config = cast(SysPearsonCorrConfig, self._get_config(config))
        scores = self.get_scores_from_stats(single_stat, config)
        keys = [i for i in scores.keys()]
        if config.no_human:
            keys = [
                key
                for key in keys
                if (
                    'Human' not in key
                    and 'HUMAN' not in key
                    and (not key.startswith('ref'))
                )
            ]
        if not config.no_human:
            keys = [key for key in keys if 'Human-A.0' not in key]

        manual_score = []
        system_score = []

        for sys_name in keys:
            if len(scores[sys_name][0]) != 0:
                manual_score.append(sum(scores[sys_name][0]) / len(scores[sys_name][0]))
            else:
                manual_score.append(0)
            if len(scores[sys_name][1]) != 0:
                system_score.append(sum(scores[sys_name][1]) / len(scores[sys_name][1]))
            else:
                manual_score.append(0)
        assert len(system_score) == len(manual_score)
        # print("number of system:"+str(len(system_score)))

        val = pearsonr(system_score, manual_score)[0]

        return val
