from __future__ import annotations

from dataclasses import dataclass
from typing import cast, Optional

import numpy as np

from explainaboard.metrics.metric import (
    AuxiliaryMetricResult,
    Metric,
    MetricConfig,
    MetricResult,
    MetricStats,
    SimpleMetricStats,
)
from explainaboard.metrics.registry import metric_config_registry
from explainaboard.utils.agreement import fleiss_kappa
from explainaboard.utils.typing_utils import unwrap

EXTERNAL_METRICS = [
    "LikertScore_fluency",
    "LikertScore_coherence",
    "LikertScore_factuality",
]
UNANNOTATED_SYMBOL = -1


@dataclass
class ExternalEvalResult(AuxiliaryMetricResult):
    agreement: float


@dataclass
@metric_config_registry.register("ExternalEvalConfig")
class ExternalEvalConfig(MetricConfig):
    aspect: str = "fluency"
    n_annotators: int = 3
    categories: int = 5
    instruction: str = "Annotation instruction"

    def to_metric(self):
        return ExternalEval(self)


class ExternalEval(Metric):
    """
    Calculates the hits metric, telling whether the predicted output is in a set of true
    outputs.
    """

    def _get_config(self, config: Optional[MetricConfig] = None) -> MetricConfig:
        """
        Get the configuration or overwritten configuration
        :param config: Optional configuration to override the default configuration
        :return: Either the default or overridden configuration
        """
        ret_config: MetricConfig = unwrap(config) if config is not None else self.config
        return ret_config

    def is_simple_average(self, stats: MetricStats):
        return False

    def calc_stats_from_external(
        self, config: Optional[MetricConfig] = None
    ) -> MetricStats:

        config = cast(ExternalEvalConfig, self._get_config(config))
        return SimpleMetricStats(config.external_stats)

    def calc_stats_from_data(
        self, true_data: list, pred_data: list, config: Optional[MetricConfig] = None
    ) -> MetricStats:
        config = cast(ExternalEvalConfig, self._get_config(config))

        if config.external_stats is not None:
            n_sample, n_annotators = config.external_stats.shape
            if n_sample != len(true_data):
                raise ValueError(
                    f"the row number of `external_stats` should be equal "
                    f"to the number of test samples: {len(true_data)}"
                )
            if n_annotators != config.n_annotators:
                raise ValueError(
                    f"the column number of `external_stats`"
                    f"should be equal to n_annotators: "
                    f"{config.n_annotators}"
                )
            return SimpleMetricStats(config.external_stats)
        else:
            # "-1" indicates samples to be evaluated
            return SimpleMetricStats(
                np.array(
                    [
                        [UNANNOTATED_SYMBOL] * config.n_annotators
                        for t, p in zip(true_data, pred_data)
                    ]
                )
            )

    def calc_agreement(self, stats: MetricStats) -> float:
        if stats.is_batched():
            raise ValueError("Unsupported for batched statistics.")

        data = stats.get_data()

        config = cast(ExternalEvalConfig, self.config)

        n_samples, n_annotators = data.shape[0], data.shape[1]
        n_categories = config.categories
        mat_kappa = np.zeros((n_samples, n_categories))

        for i in range(n_samples):
            for j in range(n_annotators):
                category_annotated = (
                    0 if data[i][j] == UNANNOTATED_SYMBOL else data[i][j]
                )
                mat_kappa[i][category_annotated] += 1

        # self.config.agreement = fleiss_kappa(mat_kappa)
        return fleiss_kappa(mat_kappa)

    def aggregate_stats(self, stats: MetricStats) -> np.ndarray:
        """
        Aggregate sufficient statistics from multiple examples into a single example
        :param stats: stats for every example
        :return: aggregated stats
        """
        data = stats.get_batch_data() if stats.is_batched() else stats.get_data()

        if data.size == 0:
            return np.array(0.0)
        else:
            # Averaging over all axes except the batch dimension.
            return np.mean(data, axis=(-1, -2))

    def evaluate_from_stats(
        self,
        stats: MetricStats,
        conf_value: Optional[float] = None,
        config: Optional[MetricConfig] = None,
    ) -> MetricResult:
        """Return an evaluation result over stats.
        :param stats: pre-computed metric stats
        :param conf_value: if set to not None, must be a number between 0 and 1,
            indicating the p-value of confidence intervals
        :param config: a configuration to over-ride the default for this object
        :return: a resulting metric value
        """
        config = self._get_config(config)
        agg_stats = self.aggregate_stats(stats)
        agreement = self.calc_agreement(stats)
        value = self.calc_metric_from_aggregate(agg_stats, config)
        conf_interval = (
            self.calc_confidence_interval(stats, conf_value) if conf_value else None
        )
        return MetricResult(
            config,
            float(value),
            conf_interval,
            conf_value,
            ExternalEvalResult(agreement),
        )
