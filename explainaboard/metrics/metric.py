"""Base classes and interfaces used to implement evaluation metrics."""

from __future__ import annotations

import abc
import copy
from dataclasses import dataclass
from typing import Any, final, Optional, TypeVar

import numpy as np
from scipy.stats import t as stats_t

from explainaboard.serialization import common_registry
from explainaboard.serialization.types import (
    Serializable,
    SerializableData,
    SerializableDataclass,
)
from explainaboard.utils.typing_utils import narrow

# Minimum sample size the central limit theorem can be applied to.
_MIN_SAMPLE_SIZE = 30


# TODO(odashi): See mypy/issues/4717
@dataclass(frozen=True)  # type: ignore
class MetricValue(Serializable, metaclass=abc.ABCMeta):
    """Abstract class of metric values.

    Each MetricValue represents a value of individual concept. Metrics may return
    multiple MetricValues to represent multiple aspects of the metric.
    """

    pass


@common_registry.register("Score")
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


@common_registry.register("ConfidenceInterval")
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
        if self.high < self.low:
            raise ValueError(
                "`high` must be greater than or equal to `low`. "
                f"high={self.high}, low={self.low}"
            )
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


@common_registry.register("MetricResult")
@final
class MetricResult(Serializable):
    """A result of computing a metric over some data."""

    _values: dict[str, MetricValue]

    def __init__(self, values: dict[str, MetricValue]) -> None:
        """Initializes MetricResult object.

        Args:
            config: Config of the Metric that calculated this result.
            values: Values calculated by the Metric.
        """
        self._values = values

    def __eq__(self, other: object) -> bool:
        """Checks the equality with another object.

        Args:
            other: An object to check the equality with self.

        Returs:
            True if other can be treated as the same as self, False otherwise.
        """
        keys = self._values.keys()
        return (
            isinstance(other, MetricResult)
            and other._values.keys() == keys
            and all(other._values[k] == self._values[k] for k in keys)
        )

    def get_value(self, cls: type[MetricValueT], name: str) -> MetricValueT:
        """Obtains a value with specific type and name.

        Args:
            cls: Subtype of MetricValue that the resulting value has to be of.
            name: Name of the value.

        Raises:
            ValueError: `name` not found, or the value is not an instance of `cls`.
        """
        value = self._values.get(name)
        if value is None:
            raise ValueError(f'MetricValue "{name}" not found.')
        if not isinstance(value, cls):
            raise ValueError(
                f'MetricValue "{name}" is not a subclass of {cls.__name__}.'
            )
        return value

    def get_value_or_none(
        self, cls: type[MetricValueT], name: str
    ) -> MetricValueT | None:
        """Obtains a value with specific type and name.

        Args:
            cls: Subtype of MetricValue that the resulting value has to be of.
            name: Name of the value.

        Returns:
            A MetricValue with `name` and `cls`, or None if such value does not exist.
        """
        try:
            return self.get_value(cls, name)
        except ValueError:
            return None

    def serialize(self) -> dict[str, SerializableData]:
        """See Serializable.serialize."""
        return {
            "values": self._values,
        }

    @classmethod
    def deserialize(cls, data: dict[str, SerializableData]) -> Serializable:
        """See Serializable.deserialize."""
        values = data["values"]

        # TODO(odashi): Implement TypeGuard.
        if not (
            isinstance(values, dict)
            and all(
                isinstance(k, str) and isinstance(v, MetricValue)
                for k, v in values.items()
            )
        ):
            raise ValueError("`values` has incompatible data.")

        return cls(values)


@dataclass
class MetricConfig(SerializableDataclass, metaclass=abc.ABCMeta):
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

    source_language: str | None = None
    target_language: str | None = None

    @abc.abstractmethod
    def to_metric(self) -> Metric:
        """Instantiate a corresponding Metric object.

        Returns:
            Instantiated Metric object.
        """
        ...

    @final
    def replace_languages(
        self, source_language: str | None, target_language: str | None
    ) -> MetricConfig:
        """Creates a new MetricConfig with specified source/target languages.

        Args:
            source_language: New source language.
            target_language: New target language.

        Returns:
            A new MetricConfig object, in which source/target_language are replaced to
            the new config, while other values are maintained.
        """
        # NOTE(odashi): Since this class can be inherited, we need to collect every
        # member not listed in this class.
        # TODO(odashi): Avoid copy.
        copied = copy.deepcopy(self)
        copied.source_language = source_language
        copied.target_language = target_language
        return copied


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


class Metric(metaclass=abc.ABCMeta):
    """A class representing an evaluation metric.

    When a subclass needs to construct a random number generator, initialize the random
    generator as follows:

    1. Invoke the `get_seed()` method to get a numpy SeedSequence;
    2. Spawn the SeedSequence with the SeedSequence's `spawn` method;
    3. Pass the spawned SeedSequence to the random generator's constructor.

    Example:
        class FooMetric(Metric):
            def foo(self) -> None:
                # Spawns a SeedSequence.
                seed = self.get_seed().spawn(1)[0]
                # Initializes a random generator with the spawned seed.
                rng = np.random.default_rng(seed)
                # Do something with `rng`.

        config = MetricConfig(...)
        metric = FooMetric(config, seed=np.random.SeedSequence(12345))
        metric.foo()
    """

    config: MetricConfig

    def __init__(
        self, config: MetricConfig, seed: np.random.SeedSequence | None = None
    ):
        """Initialize the metric.

        Args:
            config: The configuration for the metric
            seed: A seed, used to initialize a random generator in this class or
                subclasses. If None, the default seed is used.
        """
        self.config = config
        self._seed = seed if seed is not None else np.random.SeedSequence()

    @abc.abstractmethod
    def calc_stats_from_data(
        self, true_data: list[Any], pred_data: list[Any]
    ) -> MetricStats:
        """From a list of true data and predicted data, calculate sufficient statistics.

        These statistics are the numbers necessary for each data example so that the
        evaluation metric can be calculated later. In the simplest form, this is just
        the evaluation metric value for each example.

        Args:
            true_data: gold-standard data
            pred_data: predicted data

        Returns:
            A numpy array of shape [len(true_data), X] where X=1 in the simplest case of
            decomposable eval metrics
        """
        ...

    @final
    def get_seed(self) -> np.random.SeedSequence:
        """Gets a numpy SeedSequence.

        Returned SeedSequence can be used to construct a random generator in Metric
        or the subclasses. See also the `Metric` class docstring.

        Returns:
            A numpy SeedSequence.
        """
        return self._seed

    @final
    def aggregate_stats(
        self, stats: MetricStats
    ) -> np.ndarray[tuple[int], Any] | np.ndarray[tuple[int, int], Any]:
        """Aggregate sufficient statistics from multiple examples into a single example.

        Args:
            stats: stats for every example

        Returns:
            Aggregated stats. Shape must be:
                - Non-batched data: [num_aggregate_stats]
                - Batched data: [num_batches, num_aggregate_stats]
        """
        result = self._aggregate_stats(stats)

        if self.uses_customized_aggregate():
            if stats.is_batched():
                data = stats.get_batch_data()
                assert (
                    result.shape[0] == data.shape[0]
                    and result.ndim == self.stats_ndim() + 1
                ), (
                    "BUG: invalid operation: "
                    f"{type(self).__name__}._aggregate_stats(): "
                    f"Expected batch size {stats.get_batch_data().shape[0]} and "
                    f"number of dimensions {self.stats_ndim()+1}, but "
                    f"got batch size {result.shape[0]} and number of dimensions "
                    f"{result.ndim}."
                )
            else:
                assert result.ndim == self.stats_ndim(), (
                    "BUG: invalid operation: "
                    f"{type(self).__name__}._aggregate_stats(): "
                    f"Expected number of dimensions {self.stats_ndim() + 1}, but "
                    f"got number of dimensions {result.ndim}."
                )
        else:
            result_shape = (
                (stats.get_batch_data().shape[0], stats.num_statistics())
                if stats.is_batched()
                else (stats.num_statistics(),)
            )
            assert result.shape == result_shape, (
                "BUG: invalid operation: "
                f"{type(self).__name__}._aggregate_stats(): "
                f"Expected shape {result_shape}, but got {result.shape}."
            )

        return result

    def _aggregate_stats(
        self, stats: MetricStats
    ) -> np.ndarray[tuple[int], Any] | np.ndarray[tuple[int, int], Any]:
        """Inner function of aggregate_stats."""
        data = stats.get_batch_data() if stats.is_batched() else stats.get_data()
        if data.shape[-2] == 0:
            return np.zeros(
                shape=data.shape[:-2] + (data.shape[-1],),
                dtype=np.float32,
            )
        else:
            return np.mean(data, axis=-2)

    @final
    def calc_metric_from_aggregate(
        self,
        agg_stats: np.ndarray[tuple[int], Any] | np.ndarray[tuple[int, int], Any],
    ) -> np.ndarray[tuple[()], Any] | np.ndarray[tuple[int], Any]:
        """From aggregated sufficient statistics, calculate the metric value.

        Args:
            agg_stats: aggregated statistics. Shape must be:
                - Non-batched data: [num_aggregate_stats]
                - Batched data: [num_batches, num_aggregate_stats]

        Returns:
            Calculated metrics. Shape must be:
                - Non-batched data: []
                - Batched data: [num_batches]
        """
        if agg_stats.ndim not in (self.stats_ndim(), self.stats_ndim() + 1):
            raise ValueError(f"Invalid shape size: {agg_stats.shape}")

        result = self._calc_metric_from_aggregate(agg_stats)
        result_shape = (
            () if agg_stats.ndim == self.stats_ndim() else (agg_stats.shape[0],)
        )

        assert result.shape == result_shape, (
            "BUG: invalid operation: "
            f"{type(self).__name__}._calc_metric_from_aggregate(): "
            f"Expected shape {result_shape}, but got {result.shape}."
        )

        return result

    def _calc_metric_from_aggregate(
        self,
        agg_stats: np.ndarray[tuple[int], Any] | np.ndarray[tuple[int, int], Any],
    ) -> np.ndarray[tuple[()], Any] | np.ndarray[tuple[int], Any]:
        """Inner function of calc_metric_from_aggregate."""
        if agg_stats.shape[-1] != 1:
            raise ValueError(
                "Multiple aggregates can't be integrated without specific algorithms."
            )

        return agg_stats.squeeze(-1)

    def is_simple_average(self, stats: MetricStats):
        """Whether the eval score is a simple average of the sufficient statistics.

        If so the t-test is applicable, which is much more efficient. Otherwise we do
        bootstrapping to calculate confidence interval, which is slower and potentially
        less effective.
        """
        return True

    def uses_customized_aggregate(self) -> bool:
        """Whether the metric uses other aggregated stats than example-level stats.

        If this function returns True, aggregate_stats() skips some checks on the size
        of the last dimension of the returned ndarray. Note that this increases the
        possibility of implementation mistakes, and should be used with caution.
        """
        return False

    def stats_ndim(self) -> int:
        """The number of dimensions in the sufficient statistics."""
        return 1

    def calc_confidence_interval(
        self,
        stats: MetricStats,
        confidence_alpha: float,
        num_iterations: int = 1000,
    ) -> tuple[float, float] | None:
        """Calculate the confidence interval of a statistics function.

        Args:
            stats: sufficient statistics as calculated by calc_stats_from_data
            confidence_alpha: the inverse confidence level of the confidence interval
            num_iterations: the number of iterations to perform resampling

        Returns:
            A confidence interval or `None` if one cannot be calculated.
        """
        if not (0.0 < confidence_alpha < 1.0):
            raise ValueError(f"Invalid confidence_alpha: {confidence_alpha}")

        if stats.is_batched():
            raise ValueError(
                "Confidence interval can't be calculated for batched data."
            )

        stats_data = stats.get_batch_data() if stats.is_batched() else stats.get_data()
        num_stats = stats.num_statistics()
        sample_size = len(stats)

        if stats_data.shape[-2] <= 1:
            # We cannot calculate confidence intervals if we only have a single sample
            return None

        # Do t-test if applicable
        elif self.is_simple_average(stats) and sample_size >= _MIN_SAMPLE_SIZE:
            if num_stats != 1:
                raise ValueError(
                    "t-test can be applied for only 1 stat, "
                    f"but the MetricStats has {num_stats} stats."
                )
            my_mean = np.mean(stats_data)
            my_sem = np.std(stats_data) / np.sqrt(sample_size)
            if my_sem == 0.0:
                return (float(my_mean), float(my_mean))
            return stats_t.interval(
                confidence=1.0 - confidence_alpha,  # See ExplainaBoard/issues/510
                df=stats_data.shape[-2] - 1,
                loc=my_mean,
                scale=my_sem,
            )
        # Do bootstrapping otherwise
        else:
            all_indices = np.array(range(sample_size))
            rng = np.random.default_rng(self.get_seed())
            all_indices = rng.choice(
                all_indices, size=(num_iterations, sample_size), replace=True
            )
            filt_stats = stats.filter(all_indices)
            agg_stats = self.aggregate_stats(filt_stats)
            samp_results = self.calc_metric_from_aggregate(agg_stats)

            if samp_results.ndim != 1:
                raise ValueError(
                    f"Invalid shape of sampled metrics: {samp_results.shape}"
                )

            samp_results.sort()
            low = int(num_iterations * confidence_alpha / 2.0)
            high = int(num_iterations * (1.0 - confidence_alpha / 2.0))
            return float(samp_results[low]), float(samp_results[high])

    def calc_metric_from_auxiliary_stats(
        self, auxiliary_stats: MetricStats
    ) -> np.ndarray[tuple[()], Any] | np.ndarray[tuple[int], Any]:
        """Calculate the auxiliary result from auxiliary stats."""
        raise NotImplementedError

    def evaluate_from_stats(
        self,
        stats: MetricStats,
        confidence_alpha: Optional[float] = None,
        auxiliary_stats: Optional[MetricStats] = None,
    ) -> MetricResult:
        """Return an evaluation result over stats.

        Args:
            stats: pre-computed metric stats
            confidence_alpha: if set to not None, must be a number between 0 and 1,
                indicating the inverse confidence level of confidence intervals
            auxiliary_stats: metric stats used to calculate auxiliary metric result

        Returns:
            a resulting metric value
        """
        if stats.is_batched():
            raise ValueError("Batched stats can't be evaluated.")

        agg_stats = self.aggregate_stats(stats)
        score = self.calc_metric_from_aggregate(agg_stats)

        assert score.ndim == 0, "BUG: obtained batched data."

        metric_values: dict[str, MetricValue] = {
            "score": Score(float(score)),
        }

        if confidence_alpha is not None:
            ci = self.calc_confidence_interval(stats, confidence_alpha)
            if ci is not None:
                metric_values["score_ci"] = ConfidenceInterval(
                    ci[0], ci[1], confidence_alpha
                )

        return MetricResult(metric_values)

    def evaluate(
        self,
        true_data: list,
        pred_data: list,
        confidence_alpha: Optional[float] = None,
    ) -> MetricResult:
        """Return an evaluation result over true data and predicted data.

        Args:
            true_data: gold-standard data
            pred_data: predicted data
            confidence_alpha: if set to not None, must be a number between 0 and 1,
                indicating the inverse confidence level of confidence intervals

        Returns:
            a resulting metric value
        """
        stats = self.calc_stats_from_data(true_data, pred_data)
        return self.evaluate_from_stats(stats, confidence_alpha)
