from __future__ import annotations

from abc import ABCMeta, abstractmethod
from collections.abc import Callable
from typing import Any, final, TypeVar

from explainaboard.serialization.registry import TypeRegistry
from explainaboard.serialization.serializers import PrimitiveSerializer
from explainaboard.serialization.types import Serializable, SerializableData
from explainaboard.utils.logging import get_logger
from explainaboard.utils.typing_utils import narrow

_feature_type_registry = TypeRegistry[Serializable]()

T = TypeVar("T")


def get_feature_type_serializer() -> PrimitiveSerializer:
    """Returns a serializer object for FeatureTypes.

    Returns:
        A serializer object.
    """
    return PrimitiveSerializer(_feature_type_registry)


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
    def __init__(
        self,
        *,
        dtype: str | None = None,
        description: str | None = None,
        func: Callable[..., Any] | None = None,
        require_training_set: bool | None = None,
    ) -> None:
        """Initializes FeatureType object.

        Args:
            dtype: Data type specifier.
            description: Description of this feature.
            func: Function to calculate this feature from other features.
            require_training_set: Whether this feature relies on the training samples.
        """
        self._dtype = dtype
        self._description = description
        self._func = func
        self._require_training_set = (
            require_training_set if require_training_set is not None else False
        )

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
            self._dtype == other._dtype
            and self._description == other._description
            and self._func is other._func
            and self._require_training_set == other._require_training_set
        )

    @final
    @property
    def dtype(self) -> str | None:
        return self._dtype

    @final
    @property
    def description(self) -> str | None:
        return self._description

    @final
    @property
    def func(self) -> Callable[..., Any] | None:
        return self._func

    @final
    @property
    def require_training_set(self) -> bool:
        return self._require_training_set

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
            "dtype": self._dtype,
            "description": self._description,
            "require_training_set": self._require_training_set,
        }


@final
@_feature_type_registry.register("Sequence")
class Sequence(FeatureType):
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
            dtype="list",
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
@_feature_type_registry.register("Dict")
class Dict(FeatureType):
    def __init__(
        self,
        *,
        description: str | None = None,
        func: Callable[..., Any] | None = None,
        require_training_set: bool | None = None,
        feature: dict[str, FeatureType],
    ) -> None:
        """Initializes Dict object.

        Args:
            description: See FeatureType.__init__.
            func: See FeatureType.__init__.
            require_training_set: See FeatureType.__init__.
            feature: Definitions of member types.
        """
        super().__init__(
            dtype="dict",
            description=description,
            func=func,
            require_training_set=require_training_set,
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


@final
@_feature_type_registry.register("Value")
class Value(FeatureType):
    def __init__(
        self,
        *,
        dtype: str | None = None,
        description: str | None = None,
        func: Callable[..., Any] | None = None,
        require_training_set: bool | None = None,
        max_value: int | float | None = None,
        min_value: int | float | None = None,
    ) -> None:
        """Initializes Value object.

        Args:
            dtype: See FeatureType.__init__.
            description: See FeatureType.__init__.
            func: See FeatureType.__init__.
            require_training_set: See FeatureType.__init__.
            max_value: The maximum value (inclusive) of values with int/float dtype.
            min_value: The minimum value (inclusive) of values with int/float dtype.
        """
        # Fix inferred types.
        if dtype == "double":
            dtype = "float64"
        elif dtype == "float":
            dtype = "float32"

        super().__init__(
            dtype=dtype,
            description=description,
            func=func,
            require_training_set=require_training_set,
        )
        self._max_value = max_value
        self._min_value = min_value

    def __eq__(self, other: object) -> bool:
        """See FeatureType.__eq__."""
        return (
            isinstance(other, Value)
            and self._eq_base(other)
            and self._max_value == other._max_value
            and self._min_value == other._min_value
        )

    @property
    def max_value(self) -> int | float | None:
        return self._max_value

    @property
    def min_value(self) -> int | float | None:
        return self._min_value

    def serialize(self) -> dict[str, SerializableData]:
        """See Serializable.serialize."""
        data = self._serialize_base()
        data["max_value"] = self._max_value
        data["min_value"] = self._min_value
        return data

    @classmethod
    def deserialize(cls, data: dict[str, SerializableData]) -> Serializable:
        """See Serializable.deserialize."""
        max_value = data.get("max_value")
        min_value = data.get("min_value")
        if max_value is not None and not isinstance(max_value, (int, float)):
            raise ValueError(
                f"Unexpected type of `max_value`: {type(max_value).__name__}"
            )
        if min_value is not None and not isinstance(min_value, (int, float)):
            raise ValueError(
                f"Unexpected type of `min_value`: {type(min_value).__name__}"
            )

        return cls(
            dtype=_get_value(str, data, "dtype"),
            description=_get_value(str, data, "description"),
            func=None,
            require_training_set=_get_value(bool, data, "require_training_set"),
            max_value=max_value,
            min_value=min_value,
        )
