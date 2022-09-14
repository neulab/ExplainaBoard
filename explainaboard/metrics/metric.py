"""Base classes and interfaces used to implement evaluation metrics."""

from __future__ import annotations

import abc
import dataclasses
from dataclasses import dataclass
from typing import Any, final, Optional, TypeVar

import numpy as np
from scipy.stats import t as stats_t

from explainaboard.serialization.registry import TypeRegistry
from explainaboard.serialization.serializers import PrimitiveSerializer
from explainaboard.serialization.types import Serializable, SerializableData
from explainaboard.utils.typing_utils import narrow, unwrap_or

_metric_registry = TypeRegistry[Serializable]()


def get_serializer() -> PrimitiveSerializer:
    """Obtains a serializer object for this module.

    Returns:
        A serializer object.
    """
    return PrimitiveSerializer(_metric_registry)


@dataclass
class AuxiliaryMetricResult:
    """Extra information specific to individual metrics."""

    pass


# TODO(odashi): See mypy/issues/4717
@dataclass(frozen=True)  # type: ignore
class MetricValue(Serializable, metaclass=abc.ABCMeta):
    """Abstract class of metric values.

    Each MetricValue represents a value of individual concept. Metrics may return
    multiple MetricValues to represent multiple aspects of the metric.
    """

    pass


@_metric_registry.register("Score")
@final
@dataclass(frozen=True)
class Score(MetricValue):
    """MetricValue representing a real, unbound value.

    Args:
        value: The value of this object.
    """

    value: float

    def serialize(self) -> dict[str, SerializableData]:
        """See Serializable.serialize."""
        return {"value": self.value}

    @classmethod
    def deserialize(cls, data: dict[str, SerializableData]) -> Serializable:
        """See Serializable.deserialize."""
        return cls(value=narrow(float, data["value"]))


@_metric_registry.register("ConfidenceInterval")
@final
@dataclass(frozen=True)
class ConfidenceInterval(MetricValue):
    """MetricValue representing a confidence interval with its confidence level.

    Args:
        low: The lower bound of the interval.
        high: The upper bound of the interval. This value must be greater than `low`.
        alpha: The inverse confidence level of the interval. If the object represents a
            95% confidence interval, this value must be 0.05.
    """

    low: float
    high: float
    alpha: float

    def __post_init__(self) -> None:
        """Validate values of members."""
        if self.high <= self.low:
            raise ValueError("`high` must be greater than `low`.")
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("`alpha` must be in between 0.0 and 1.0.")

    def serialize(self) -> dict[str, SerializableData]:
        """See Serializable.serialize."""
        return {
            "low": self.low,
            "high": self.high,
            "alpha": self.alpha,
        }

    @classmethod
    def deserialize(cls, data: dict[str, SerializableData]) -> Serializable:
        """See Serializable.deserialize."""
        return cls(
            low=narrow(float, data["low"]),
            high=narrow(float, data["high"]),
            alpha=narrow(float, data["alpha"]),
        )


MetricValueT = TypeVar("MetricValueT", bound=MetricValue)


@_metric_registry.register("MetricResult")
@final
class MetricResult(Serializable):
    """A result of computing a metric over some data."""

    _config: MetricConfig
    _values: dict[str, MetricValue]

    def __init__(self, config: MetricConfig, values: dict[str, MetricValue]) -> None:
        """Initializes MetricResult object.

        Args:
            config: Config of the Metric that calculated this result.
            values: Values calculated by the Metric.
        """
        self._config = config
        self._values = values

    @property
    def config(self) -> MetricConfig:
        """Obtains underlying MetricConfig.

        Returns:
            A MetricConfig object related to this MetricResult.
        """
        return self._config

    def get_value(self, cls: type[MetricValueT], name: str) -> MetricValueT | None:
        """Obtains a value with specific type and name.

        Args:
            cls: Subtype of MetricValue that the resulting value has to be of.
            name: Name of the value.

        Returns:
            A MetricValue with `name` and `cls`, or None if such value does not exist.
        """
        value = self._values.get(name)
        if value is None:
            return None
        return value if isinstance(value, cls) else None

    def serialize(self) -> dict[str, SerializableData]:
        """See Serializable.serialize."""
        return {
            "config": dataclasses.asdict(self.config),
            "values": self._values,
        }

    @classmethod
    def deserialize(cls, data: dict[str, SerializableData]) -> Serializable:
        """See Serializable.deserialize."""
        # TODO(odahsi): implement Serializable interface for MetricConfig.
        raise NotImplementedError("MetricResult can't be deserialized at this point.")


@dataclass
class MetricConfig(dict):
    """The configuration for a metric.

    This can be passed in to the metric either in
    the constructor (e.g. for compute-intensive operations such as model loading),
    or when performing individual metric computation.

    Args:
        name: The metric name
        source_language: The source language
        target_language: The target language
        cls_name: The class name
    """

    name: str
    source_language: str | None = None
    target_language: str | None = None
    cls_name: str | None = None

    def __post_init__(self) -> None:
        """Set the class name for the metric config."""
        self.cls_name = type(self).__name__

    def to_metric(self) -> Metric:
        """See MetricConfig.to_metric."""
        raise NotImplementedError

    @classmethod
    def dict_conv(cls, k: str, v: Any) -> Any:
        """Conversion for serialization."""
        return v


class MetricStats(metaclass=abc.ABCMeta):
    """Interface of sufficient statistics necessary to calculate a metric."""

    @abc.abstractmethod
    def __len__(self) -> int:
        """Returns the "length" of the dataset.

        Returns:
            The "length". It must be either:
            - Number of the whole samples if `is_batched() == False`
            - Number of batches if `is_batched() == True`
        """
        ...

    @abc.abstractmethod
    def is_batched(self) -> bool:
        """Returns whether this statistics object is batched or not.

        If this function returns True, `get_batch_data()` must return a corresponding
        value.

        Returns:
            `True` if the underlying data is batched, `False` otherwise.
        """
        ...

    @abc.abstractmethod
    def num_statistics(self) -> int:
        """Returns the number of statistics for each data.

        This value must be the same as the size of the last dimension in `get_data()`.

        Returns:
            The number of statistics for each data.
        """
        ...

    @abc.abstractmethod
    def get_data(self) -> np.ndarray[tuple[int, int], Any]:
        """Get the sufficient statistics in ndarray format.

        This function must always return a 2-dimensional ndarray.
        This function may return a shallow copy of the underlying object. Changing the
        return value in-place may cause unintended changes of the behavior.

        Returns:
            The sufficient statistics.
            The shape must be `[dataset_length, num_statistics]`.
        """
        ...

    @abc.abstractmethod
    def get_batch_data(self) -> np.ndarray[tuple[int, int, int], Any]:
        """Get the sufficient statistics in ndarray format.

        This function must always return a 3-dimensional ndarray.
        This function may return a shallow copy of the underlying object. Changing the
        return value in-place may cause unintended changes of the behavior.

        Returns:
            The sufficient statistics.
            The shape must be `[num_batches, batch_size, num_statistics]`.
        """
        ...

    @final
    def filter(self, indices: list[int] | np.ndarray) -> MetricStats:
        """Return a view of these stats filtered down to the indicated indices.

        This function requires that the statistics is not batched.

        This function may return a shallow copy of the underlying object. Changing the
        return value in-place may cause unintended changes of the behavior.

        Args:
            indices: The indices over which the stats should be calculated.
                Shape must be either:
                - `[num_indices]` to simply filter the whole data.
                - `[num_batches, num_indices]` to filter the whole data and make the
                    batched results.

        Returns:
            The filtered statistics.
            If the given indices is 1-dimensional, it is non-batched statistics.
            Otherwise, it is batched statistics.

        Raises:
            ValueError: Attempted unsupported operation.
        """
        if self.is_batched():
            raise ValueError("MetricStats.filter() does not support batched data.")

        indices_array = np.array(indices)
        if indices_array.ndim not in [1, 2]:
            raise ValueError(f"Unsupported shape: {indices_array.shape}")

        data = self.get_data()
        filtered = data[indices_array.flatten()]
        filtered_batched = filtered.reshape(indices_array.shape + (data.shape[1],))
        return SimpleMetricStats(filtered_batched)


@final
class SimpleMetricStats(MetricStats):
    """MetricStats that directly holds an ndarray.

    This class may hold a shallow copy of the given ndarray.
    In-place modification of the array results in unexpected change of the behavior.
    """

    def __init__(self, data: np.ndarray) -> None:
        """Initializes SimpleMetricsStats.

        Args:
            data: A numpy array representing the statistics.
                The shape must be either:
                - `[dataset_length]` for representing the whole data with 1 value.
                - `[dataset_length, num_statistics]` for representing the whole data
                - `[num_batches, batch_size, num_statistics]` for representing batched
                    data.

        Rasies:
            ValueError: The given data has an unsupported shape.
        """
        if data.ndim == 1:
            self._data = np.expand_dims(data, 1)
        elif data.ndim in [2, 3]:
            self._data = data
        else:
            raise ValueError(f"data has unsupported shape: {data.shape}")

        # The shape size must be either 2 or 3 at this point.
        self._batched = self._data.ndim == 3

    def __len__(self) -> int:
        """See MetricStats.__len__."""
        return len(self._data)

    def is_batched(self) -> bool:
        """See MetricStats.is_batched."""
        return self._batched

    def num_statistics(self) -> int:
        """See MetricStats.num_statistics."""
        return self._data.shape[-1]

    def get_data(self) -> np.ndarray[tuple[int, int], Any]:
        """See MetricStats.get_data."""
        if self.is_batched():
            raise RuntimeError(
                "Attempted to obtain non-batched data from batched Statistics."
            )
        return self._data

    def get_batch_data(self) -> np.ndarray[tuple[int, int, int], Any]:
        """See MetricStats.get_batch_data."""
        if not self.is_batched():
            raise RuntimeError(
                "Attempted to obtain batched data from non-batched Statistics."
            )
        return self._data


class Metric:
    """A class representing an evaluation metric."""

    def __init__(
        self,
        config: MetricConfig,
    ):
        """Initialize the metric.

        :param config: The configuration for the metric
        """
        self.config: MetricConfig = config

    @abc.abstractmethod
    def calc_stats_from_data(
        self, true_data: list, pred_data: list, config: Optional[MetricConfig] = None
    ) -> MetricStats:
        """From a list of true data and predicted data, calculate sufficient statistics.

        These statistics are the numbers necessary for each data example so that the
        evaluation metric can be calculated later. In the simplest form, this is just
        the evaluation metric value for each example.

        :param true_data: gold-standard data
        :param pred_data: predicted data
        :param config: a configuration to over-ride the default for this object
        :return: a numpy array of shape [len(true_data), X] where X=1 in the simplest
            case of decomposable eval metrics
        """
        ...

    def aggregate_stats(self, stats: MetricStats) -> np.ndarray:
        """Aggregate sufficient statistics from multiple examples into a single example.

        Args:
            stats: stats for every example

        Returns:
            Aggregated stats
        """
        data = stats.get_batch_data() if stats.is_batched() else stats.get_data()
        if data.size == 0:
            return np.array(0.0)
        else:
            return np.mean(data, axis=-2)

    def calc_metric_from_aggregate(
        self, agg_stats: np.ndarray, config: Optional[MetricConfig] = None
    ) -> np.ndarray:
        """From aggregated sufficient statistics, calculate the metric value.

        Args:
            agg_stats: aggregated statistics, either:
                one-dimensional [metric_size]
                two-dimensional [batch_size, metric_size]
            config: a configuration to over-ride the default for this object

        Returns:
            calculated metric of size 1, or metrics of size [batch_size]
        """
        return agg_stats

    def is_simple_average(self, stats: MetricStats):
        """Whether the eval score is a simple average of the sufficient statistics.

        If so the t-test is applicable, which is much more efficient. Otherwise we do
        bootstrapping to calculate confidence interval, which is slower and potentially
        less effective.
        """
        return True

    def calc_confidence_interval(
        self,
        stats: MetricStats,
        confidence_alpha: float,
        n_samples: int = 1000,
        prop_samples: float = 0.5,
        config: Optional[MetricConfig] = None,
    ) -> tuple[float, float] | None:
        """Calculate the confidence interval of a statistics function.

        Args:
            stats: sufficient statistics as calculated by calc_stats_from_data
            confidence_alpha: the inverse confidence level of the confidence interval
            n_samples: the number of bootstrapping samples
            prop_samples: the proportion of samples to sample each time
            config: a configuration to over-ride the default for this object

        Returns:
            A confidence interval or `None` if one cannot be calculated.
        """
        if confidence_alpha <= 0.0 or confidence_alpha >= 1.0:
            raise ValueError(f'Bad confidence value {confidence_alpha}')

        stats_data = stats.get_batch_data() if stats.is_batched() else stats.get_data()
        num_stats = stats.num_statistics()

        if stats_data.shape[-2] <= 1:
            # We cannot calculate confidence intervals if we only have a single sample
            return None

        # Do t-test if applicable
        elif self.is_simple_average(stats):
            if num_stats != 1:
                raise ValueError(
                    "t-test can be applied for only 1 stat, "
                    f"but the MetricStats has {num_stats} stats."
                )
            my_mean = np.mean(stats_data)
            my_std = np.std(stats_data)
            if my_std == 0.0:
                return (float(my_mean), float(my_mean))
            return stats_t.interval(
                alpha=confidence_alpha,
                df=stats_data.shape[-2] - 1,
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
            low = int(n_samples * confidence_alpha / 2.0)
            high = int(n_samples * (1.0 - confidence_alpha / 2.0))
            return float(samp_results[low]), float(samp_results[high])

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
                indicating the inverse confidence level of confidence intervals
            config: a configuration to over-ride the default for this object

        Returns:
            a resulting metric value
        """
        actual_config = unwrap_or(config, self.config)
        agg_stats = self.aggregate_stats(stats)

        metric_values: dict[str, MetricValue] = {
            "score": Score(self.calc_metric_from_aggregate(agg_stats, actual_config))
        }

        if confidence_alpha is not None:
            ci = self.calc_confidence_interval(stats, confidence_alpha)
            if ci is not None:
                metric_values["score_ci"] = ConfidenceInterval(
                    ci[0], ci[1], confidence_alpha
                )

        return MetricResult(actual_config, metric_values)

    def evaluate(
        self,
        true_data: list,
        pred_data: list,
        confidence_alpha: Optional[float] = None,
        config: Optional[MetricConfig] = None,
    ) -> MetricResult:
        """Return an evaluation result over true data and predicted data.

        Args:
            true_data: gold-standard data
            pred_data: predicted data
            confidence_alpha: if set to not None, must be a number between 0 and 1,
                indicating the inverse confidence level of confidence intervals
            config: a configuration to over-ride the default for this object

        Returns:
            a resulting metric value
        """
        stats = self.calc_stats_from_data(true_data, pred_data, config)
        return self.evaluate_from_stats(stats, confidence_alpha, config)
