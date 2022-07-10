from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.stats import pearsonr
from explainaboard.metrics.metric import Metric, MetricConfig, MetricStats
from explainaboard.metrics.registry import register_metric_config
from typing import cast, Optional, Union

@dataclass
@register_metric_config
class SegKtauCorrConfig(MetricConfig):
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
                    'Human' in sys_name or 'HUMAN' in sys_name or sys_name.startswith(
                'ref')
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
                if (
                        abs(score[i][0] - score[j][0]) < config.threshold
                        or abs(score[i][0] - score[j][0]) == 0
                ):
                    continue
                elif (
                        score[i][0] - score[j][0] >= config.threshold
                ):  # system i is better than system j
                    if score[i][1] > score[j][1]:
                        conc += 1
                    else:
                        disc += 1
                    num += 1
                else:  # system i is worse than system j
                    if score[i][1] < score[j][1]:
                        conc += 1
                    else:
                        disc += 1
                    num += 1
        return conc, disc, num

    def calc_metric_from_aggregate(
            self, agg_stats: np.ndarray, config: Optional[MetricConfig] = None
    ) -> float:
        scores = self.get_scores_from_stats(agg_stats, config)
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
                    'Human' in sys_name or 'HUMAN' in sys_name or sys_name.startswith(
                'ref')
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
        score_manuals, scores = self.get_scores_from_stats(agg_stats, config)
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
        config = cast(SysPearsonCorrConfig, self._get_config(config))
        scores = self.get_scores_from_stats(agg_stats, config)

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



