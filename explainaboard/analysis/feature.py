"""Classes to express features."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from collections.abc import Callable
from enum import Enum
from typing import Any, final, TypeVar

from explainaboard.serialization import common_registry
from explainaboard.serialization.types import Serializable, SerializableData
from explainaboard.utils.logging import get_logger
from explainaboard.utils.typing_utils import narrow

T = TypeVar("T")


def _get_value(cls: type[T], data: dict[str, SerializableData], key: str) -> T | None:
    """Helper to obtain typed value in the SerializableData dict.

    Args:
        cls: Type to obtain.
        data: Dict containing the target value.
        key: Key of the target value.

    Returs:
        Typed target value, or None if it does not exist.
    """
    value = data.get(key)
    return narrow(cls, value) if value is not None else None


class FeatureType(Serializable, metaclass=ABCMeta):
    """An object specifying the type of features."""

    def __init__(
        self,
        *,
        description: str | None = None,
        func: Callable[..., Any] | None = None,
        require_training_set: bool | None = None,
        optional: bool = False,
    ) -> None:
        """Initializes FeatureType object.

        Args:
            description: Description of this feature.
            func: Function to calculate this feature from other features.
            require_training_set: Whether this feature relies on the training samples.
            optional: set it to True if this feature is optional.
        """
        self._description = description
        self._func = func
        self._require_training_set = (
            require_training_set if require_training_set is not None else False
        )
        self._optional = optional

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        """Checks if two FeatureTypes are the same.

        Args:
            other: FeatureType to compare.

        Returns:
            True if `other` can be treated as the same value with `self`, False
            otherwise.
        """
        ...

    @final
    def _eq_base(self, other: FeatureType) -> bool:
        """Helper to compare two FeatureTypes have the same base members.

        Args:
            other: FeatureType to compare.

        Returns:
            True if `other` has the same base members with `self`, False otherwise.
        """
        return (
            self._description == other._description
            and self._func is other._func
            and self._require_training_set == other._require_training_set
        )

    @final
    @property
    def description(self) -> str | None:
        """Returns the description of this feature."""
        return self._description

    @final
    @property
    def func(self) -> Callable[..., Any] | None:
        """Returns the callable to calculate this feature."""
        return self._func

    @final
    @property
    def require_training_set(self) -> bool:
        """Returns whether this feature requires training set or not."""
        return self._require_training_set

    @final
    @property
    def optional(self) -> bool:
        """Returns whether this feature is optional."""
        return self._optional

    @final
    def _serialize_base(self) -> dict[str, SerializableData]:
        """Helper to serialize base members.

        Returns:
            Serialized object containing base members.
        """
        if self.func is not None:
            # TODO(odashi): FeatureTypes with `func` can't be restored correctly from
            # the serialized data. If you met this warning, it seems there could be
            # potential bugs.
            # Remove `func` member from FeatureType to correctly serialize these
            # objects.
            get_logger(__name__).warning("`func` member is not serializable.")

        return {
            "description": self._description,
            "require_training_set": self._require_training_set,
        }


@final
@common_registry.register("Sequence")
class Sequence(FeatureType):
    """A feature consisting of a sequence of features."""

    def __init__(
        self,
        *,
        description: str | None = None,
        func: Callable[..., Any] | None = None,
        require_training_set: bool | None = None,
        feature: FeatureType,
    ) -> None:
        """Initializes Sequence object.

        Args:
            description: See FeatureType.__init__.
            func: See FeatureType.__init__.
            require_training_set: See FeatureType.__init__.
            feature: Feature type of elements.
        """
        super().__init__(
            description=description,
            func=func,
            require_training_set=require_training_set,
        )
        self._feature = feature

    def __eq__(self, other: object) -> bool:
        """See FeatureType.__eq__."""
        return (
            isinstance(other, Sequence)
            and self._eq_base(other)
            and self._feature == other._feature
        )

    @property
    def feature(self) -> FeatureType:
        """Returns the element type of this sequence."""
        return self._feature

    def serialize(self) -> dict[str, SerializableData]:
        """See Serializable.serialize."""
        data = self._serialize_base()
        data["feature"] = self._feature
        return data

    @classmethod
    def deserialize(cls, data: dict[str, SerializableData]) -> Serializable:
        """See Serializable.deserialize."""
        return cls(
            description=_get_value(str, data, "description"),
            func=None,
            require_training_set=_get_value(bool, data, "require_training_set"),
            # See https://github.com/python/mypy/issues/4717
            feature=narrow(FeatureType, data["feature"]),  # type: ignore
        )


@final
@common_registry.register("Dict")
class Dict(FeatureType):
    """A feature that consists of a dictionary of features."""

    def __init__(
        self,
        *,
        description: str | None = None,
        func: Callable[..., Any] | None = None,
        require_training_set: bool | None = None,
        feature: dict[str, FeatureType],
        optional: bool = False,
    ) -> None:
        """Initializes Dict object.

        Args:
            description: See FeatureType.__init__.
            func: See FeatureType.__init__.
            require_training_set: See FeatureType.__init__.
            feature: Definitions of member types.
            optional: See FeatureType.__init__.
        """
        super().__init__(
            description=description,
            func=func,
            require_training_set=require_training_set,
            optional=optional,
        )
        self._feature = feature

    def __eq__(self, other: object) -> bool:
        """See FeatureType.__eq__."""
        return (
            isinstance(other, Dict)
            and self._eq_base(other)
            and self._feature == other._feature
        )

    @property
    def feature(self) -> dict[str, FeatureType]:
        """Returns the types of underlying members in this dict."""
        return self._feature

    def serialize(self) -> dict[str, SerializableData]:
        """See Serializable.serialize."""
        data = self._serialize_base()
        data["feature"] = self._feature
        return data

    @classmethod
    def deserialize(cls, data: dict[str, SerializableData]) -> Serializable:
        """See Serializable.deserialize."""
        feature = {
            # See https://github.com/python/mypy/issues/4717
            k: narrow(FeatureType, v)  # type: ignore
            for k, v in narrow(dict, data["feature"]).items()
        }

        return cls(
            description=_get_value(str, data, "description"),
            func=None,
            require_training_set=_get_value(bool, data, "require_training_set"),
            feature=feature,
        )


# TODO(odashi): Follow well-known schema to define this struct, e.g., JSON Schema.
@final
class DataType(Enum):
    """Data types for Value FeatureType."""

    INT = "int"
    FLOAT = "float"
    STRING = "string"


@final
@common_registry.register("Value")
class Value(FeatureType):
    """A feature representing a value such as an int, float, or string."""

    _dtype: DataType
    _max_value: int | float | None
    _min_value: int | float | None

    def __init__(
        self,
        *,
        dtype: DataType,
        description: str | None = None,
        func: Callable[..., Any] | None = None,
        require_training_set: bool | None = None,
        max_value: int | float | None = None,
        min_value: int | float | None = None,
        optional: bool = False,
    ) -> None:
        """Initializes Value object.

        Args:
            dtype: Data type of this value.
            description: See FeatureType.__init__.
            func: See FeatureType.__init__.
            require_training_set: See FeatureType.__init__.
            max_value: The maximum value (inclusive) of values with int/float dtype.
            min_value: The minimum value (inclusive) of values with int/float dtype.
            optional: See FeatureType.__init__.
        """
        super().__init__(
            description=description,
            func=func,
            require_training_set=require_training_set,
            optional=optional,
        )

        self._dtype = dtype

        if max_value is not None and min_value is not None and max_value < min_value:
            raise ValueError("max_value must be greater than or equal to min_value.")

        if self._dtype == DataType.INT:
            if isinstance(max_value, float):
                raise ValueError("max_value must be an int when the dtype is integer.")
            if isinstance(min_value, float):
                raise ValueError("min_value must be an int when the dtype is integer.")
            self._max_value = max_value
            self._min_value = min_value
        elif self._dtype == DataType.FLOAT:
            self._max_value = float(max_value) if max_value is not None else None
            self._min_value = float(min_value) if min_value is not None else None
        else:
            if max_value is not None:
                raise ValueError(
                    "max_value must not be specified when dtype is not a numeric type."
                )
            if min_value is not None:
                raise ValueError(
                    "min_value must not be specified when dtype is not a numeric type."
                )
            self._max_value = max_value
            self._min_value = min_value

    def __eq__(self, other: object) -> bool:
        """See FeatureType.__eq__."""
        return (
            isinstance(other, Value)
            and self._eq_base(other)
            and self._dtype == other._dtype
            and self._max_value == other._max_value
            and self._min_value == other._min_value
        )

    @property
    def dtype(self) -> DataType:
        """Returns the data type of this value."""
        return self._dtype

    @property
    def max_value(self) -> int | float | None:
        """Returns the maximum value (inclusive) of this value."""
        return self._max_value

    @property
    def min_value(self) -> int | float | None:
        """Returns the minimum value (inclusive) of this value."""
        return self._min_value

    def serialize(self) -> dict[str, SerializableData]:
        """See Serializable.serialize."""
        data = self._serialize_base()
        data.update(
            {
                "dtype": str(self._dtype.value),
                "max_value": self._max_value,
                "min_value": self._min_value,
            }
        )
        return data

    @classmethod
    def deserialize(cls, data: dict[str, SerializableData]) -> Serializable:
        """See Serializable.deserialize."""
        max_value = data.get("max_value")
        min_value = data.get("min_value")

        if max_value is not None and not isinstance(max_value, (int, float)):
            raise ValueError("max_value must be either int, float, or None.")
        if min_value is not None and not isinstance(min_value, (int, float)):
            raise ValueError("min_value must be either int, float, or None.")

        return cls(
            dtype=DataType(data["dtype"]),
            description=_get_value(str, data, "description"),
            func=None,
            require_training_set=_get_value(bool, data, "require_training_set"),
            max_value=max_value,
            min_value=min_value,
        )
