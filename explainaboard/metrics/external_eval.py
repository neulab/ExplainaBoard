"""Evaluation metrics with externally provided statistics."""

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

UNANNOTATED_SYMBOL = -1


@dataclass
class ExternalEvalResult(AuxiliaryMetricResult):
    """The result of an external evaluation metric.

    Args:
        agreement: The agreement according to some measure (e.g. Fleiss's Kappa).
    """

    agreement: float


@dataclass
@metric_config_registry.register("ExternalEvalConfig")
class ExternalEvalConfig(MetricConfig):
    """Configuration for ExternalEval.

    Args:
        aspect: The name of the external score being calculated.
        n_annotators: The number of annotators doing annotation (in the case of human
                      human annotators).
        categories: The number of categories in the human evaluation.
        instruction: The instructions given to the annotators.
    """

    aspect: str = "fluency"
    n_annotators: int = 3
    categories: int = 5
    instruction: str = "Annotation instruction"
    # The external statistics for metrics. The row size is equal to the number
    # of test samples, the column size is equal to `n_annotators`.
    external_stats: np.ndarray | None = None

    def to_metric(self) -> Metric:
        """See MetricConfig.to_metric."""
        return ExternalEval(self)


class ExternalEval(Metric):
    """Calculates the Hits metric.

    This tells whether the predicted output is in a set of true outputs.
    """

    def _get_config(self, config: Optional[MetricConfig] = None) -> MetricConfig:
        """Get the configuration or overwritten configuration.

        Args:
            config: Optional configuration to override the default configuration

        Returns:
            Either the default or overridden configuration
        """
        ret_config: MetricConfig = unwrap(config) if config is not None else self.config
        return ret_config

    def is_simple_average(self, stats: MetricStats):
        """See Metric.is_simple_average."""
        return False

    def calc_stats_from_external(
        self, config: Optional[MetricConfig] = None
    ) -> MetricStats:
        """Calculate statistics from external data.

        Args:
            config: The configuration under which to calculate the statistics.

        Returns:
            The calculated statistics.
        """
        config = cast(ExternalEvalConfig, self._get_config(config))
        return SimpleMetricStats(config.external_stats)

    def calc_stats_from_data(
        self, true_data: list, pred_data: list, config: Optional[MetricConfig] = None
    ) -> MetricStats:
        """See Metric.calc_stats_from_data."""
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
        """Calculate the agreement between annotators in metric statistics.

        Args:
            stats: The statistics to calculate over.

        Returns:
            Fleiss's Kappa agreement statistic.
        """
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
        """See Metric.aggregate_stats."""
        data = stats.get_batch_data() if stats.is_batched() else stats.get_data()

        if data.size == 0:
            return np.array(0.0)
        else:
            # Averaging over all axes except the batch dimension.
            return np.mean(data, axis=(-1, -2))

    def evaluate_from_stats(
        self,
        stats: MetricStats,
        confidence_alpha: Optional[float] = None,
        config: Optional[MetricConfig] = None,
    ) -> MetricResult:
        """Return an evaluation result over stats.

        Args:
            stats: pre-computed metric stats
            confidence_alpha: if set to not None, must be a number between 0 and 1,
                indicating the inverse confidence level of the confidence interval
            config: a configuration to over-ride the default for this object

        Returns:
            a resulting metric value
        """
        config = self._get_config(config)
        agg_stats = self.aggregate_stats(stats)
        agreement = self.calc_agreement(stats)
        value = self.calc_metric_from_aggregate(agg_stats, config)
        confidence_interval = (
            self.calc_confidence_interval(stats, confidence_alpha)
            if confidence_alpha
            else None
        )
        return MetricResult(
            config,
            float(value),
            confidence_interval,
            confidence_alpha,
            ExternalEvalResult(agreement),
        )
