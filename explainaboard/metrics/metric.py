from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from scipy.stats import t as stats_t

from explainaboard.utils.typing_utils import unwrap, unwrap_or


@dataclass
class AuxiliaryMetricResult:
    pass


@dataclass
class MetricResult:
    """
    A result of computing a metric over some data
    """

    # Configuration with which it was calculated
    config: MetricConfig
    # Metric value
    value: float
    # Confidence interval of the metric values
    conf_interval: Optional[tuple[float, float]] = None
    # The p-value of the confidence interval
    conf_value: Optional[float] = None
    # Extra data for
    auxiliary_result: AuxiliaryMetricResult | None = None

    def to_dict(self):
        ret = {
            'config': self.config.__dict__,
            'value': self.value,
        }
        if self.conf_interval is not None:
            ret['conf_interval'] = self.conf_interval
        if self.conf_value is not None:
            ret['conf_value'] = self.conf_value
        return ret


@dataclass
class MetricConfig(dict):
    """
    The configuration for the metric. This can be passed in to the metric either in
    the constructor (e.g. for compute-intensive operations such as model loading),
    or when performing individual metric computation.
    """

    name: str
    source_language: str | None = None
    target_language: str | None = None
    cls_name: str | None = None
    # The external statistics for metrics
    external_stats: np.ndarray | None = None

    def __post_init__(self):
        # Save the class name
        self.cls_name = type(self).__name__

    def to_metric(self):
        raise NotImplementedError

    @classmethod
    def dict_conv(cls, k, v):
        return v


class MetricStats:
    """
    A class holding the sufficient statistics necessary to calculate a metric
    """

    def __init__(self, data: Optional[np.ndarray]):
        """
        :param data: A numpy array of dimensions [x,y], where x in the length of the
            dataset, and y is the size of the sufficient statistics necessary to
            calculate the metric. Alternatively, it can be [b,x,y] where b is the
            batch size, particularly for bootstrap sampling.
        """
        if data is not None and data.ndim == 1:
            data = data.reshape((data.shape[0], 1))
        self._data = data

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset
        """
        return len(unwrap(self._data))

    def get_data(self) -> np.ndarray:
        """
        Get the sufficient statistics in ndarray format
        """
        return unwrap(self._data)

    def filter(self, indices: Union[list[int], np.ndarray]) -> MetricStats:
        """
        Return a view of these stats filtered down to the indicated indices.
        :param indices: The indices over which the stats should be calculated
        :return: The filtered stats
        """
        sdata = self.get_data()
        if sdata.ndim != 2:
            raise ValueError(f'Can only filter non-batched statistics {sdata.shape}')
        if type(indices) != np.ndarray:
            indices = np.array(indices)
        if indices.ndim == 1:
            return MetricStats(sdata[indices])
        else:
            batch, samples = indices.shape
            indices = indices.reshape((batch * samples,))
            filtered_data = sdata[indices]
            filtered_data = filtered_data.reshape((batch, samples, sdata.shape[1]))
            return MetricStats(filtered_data)


class Metric:
    """
    A class representing an evaluation metric
    """

    def __init__(
        self,
        config: MetricConfig,
    ):
        """
        Initialize the metric
        :param config: The configuration for the metric
        """
        self.config: MetricConfig = config

    @abc.abstractmethod
    def calc_stats_from_data(
        self, true_data: list, pred_data: list, config: Optional[MetricConfig] = None
    ) -> MetricStats:
        """From a list of true data and predicted data, calculate the sufficient
        statistics for each data example so that the evaluation metric can be calculated
        later. In the simplest form, this is just the evaluation metric value for each
        example.
        :param true_data: gold-standard data
        :param pred_data: predicted data
        :param config: a configuration to over-ride the default for this object
        :return: a numpy array of shape [len(true_data), X] where X=1 in the simplest
            case of decomposable eval metrics
        """
        ...

    def aggregate_stats(self, stats: MetricStats) -> np.ndarray:
        """
        Aggregate sufficient statistics from multiple examples into a single example
        :param stats: stats for every example
        :return: aggregated stats
        """

        data = stats.get_data()
        if data.size == 0:
            return np.array(0.0)
        else:
            return np.mean(data, axis=-2)

    def calc_metric_from_aggregate(
        self, agg_stats: np.ndarray, config: Optional[MetricConfig] = None
    ) -> np.ndarray:
        """From aggregated sufficient statistics, calculate the metric value
        :param agg_stats: aggregated statistics, either:
          one-dimensional [metric_size]
          two-dimensional [batch_size, metric_size]
        :param config: a configuration to over-ride the default for this object
        :return: calculated metric of size 1, or metrics of size [batch_size]
        """
        return agg_stats

    def is_simple_average(self, stats: MetricStats):
        """
        Whether the evaluation score is a simple average of the sufficient statistics.
        If so the t-test is applicable, which is much more efficient. Otherwise we do
        bootstrapping to calculate confidence interval, which is slower and potentially
        less effective.
        """
        return True

    def calc_confidence_interval(
        self,
        stats: MetricStats,
        conf_value: float,
        n_samples: int = 1000,
        prop_samples: float = 0.5,
        config: Optional[MetricConfig] = None,
    ) -> tuple[float, float] | None:
        """
        :param stats: sufficient statistics as calculated by calc_stats_from_data
        :param conf_value: the p-value of the interval
        :param n_samples: the number of bootstrapping samples
        :param prop_samples: the proportion of samples to sample each time
        :param config: a configuration to over-ride the default for this object
        """
        if conf_value <= 0.0 or conf_value >= 1.0:
            raise ValueError(f'Bad confidence value {conf_value}')

        stats_data = stats.get_data()
        # We cannot calculate confidence intervals if we only have a single sample
        if stats_data.shape[0] <= 1:
            return None
        # Do t-test if applicable
        elif self.is_simple_average(stats):
            if stats_data.shape[1] != 1:
                raise ValueError(f'problem with shape in t-test {stats_data.shape}')
            my_mean = np.mean(stats_data)
            my_std = np.std(stats_data)
            if my_std == 0.0:
                return (float(my_mean), float(my_mean))
            return stats_t.interval(
                alpha=conf_value,
                df=stats_data.shape[0] - 1,
                loc=my_mean,
                scale=my_std,
            )
        # Do bootstrapping otherwise
        else:
            n_elems = max(int(prop_samples * len(stats)), 1)
            all_indices = np.array(range(len(stats)))
            rng = np.random.default_rng()
            all_indices = rng.choice(
                all_indices, size=(n_samples, n_elems), replace=True
            )
            filt_stats = stats.filter(all_indices)
            agg_stats = self.aggregate_stats(filt_stats)
            samp_results = self.calc_metric_from_aggregate(agg_stats, config)
            samp_results.sort()
            low = int(n_samples * conf_value / 2.0)
            high = int(n_samples * (1.0 - conf_value / 2.0))
            return float(samp_results[low]), float(samp_results[high])

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
        actual_config = unwrap_or(config, self.config)
        agg_stats = self.aggregate_stats(stats)
        value = self.calc_metric_from_aggregate(agg_stats, actual_config)
        conf_interval = (
            self.calc_confidence_interval(stats, conf_value) if conf_value else None
        )
        return MetricResult(actual_config, float(value), conf_interval, conf_value)

    def evaluate(
        self,
        true_data: list,
        pred_data: list,
        conf_value: Optional[float] = None,
        config: Optional[MetricConfig] = None,
    ) -> MetricResult:
        """Return an evaluation result over true data and predicted data.
        :param true_data: gold-standard data
        :param pred_data: predicted data
        :param conf_value: if set to not None, must be a number between 0 and 1,
            indicating the p-value of confidence intervals
        :param config: a configuration to over-ride the default for this object
        :return: a resulting metric value
        """
        stats = self.calc_stats_from_data(true_data, pred_data, config)
        return self.evaluate_from_stats(stats, conf_value, config)
