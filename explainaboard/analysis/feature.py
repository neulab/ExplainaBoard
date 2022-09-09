from __future__ import annotations

from abc import ABCMeta
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

    @property
    def dtype(self) -> str | None:
        return self._dtype

    @property
    def description(self) -> str | None:
        return self._description

    @property
    def func(self) -> Callable[..., Any] | None:
        return self._func

    @property
    def require_training_set(self) -> bool:
        return self._require_training_set

    def serialize(self) -> dict[str, SerializableData]:
        """See Serializable.serialize."""
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
        feature: FeatureType,
        description: str | None = None,
        func: Callable[..., Any] | None = None,
        require_training_set: bool | None = None,
    ) -> None:
        """Initializes Sequence object.

        Args:
            feature: Feature type of elements.
            description: See FeatureType.__init__.
            func: See FeatureType.__init__.
            require_training_set: See FeatureType.__init__.
        """
        super().__init__("list", description, func, require_training_set)
        self._feature = feature

    @property
    def feature(self) -> FeatureType:
        return self._feature

    def serialize(self) -> dict[str, SerializableData]:
        """See Serializable.serialize."""
        data = super().serialize()
        data["feature"] = self._feature
        return data

    @classmethod
    def deserialize(cls, data: dict[str, SerializableData]) -> Serializable:
        """See Serializable.deserialize."""
        return cls(
            # See https://github.com/python/mypy/issues/4717
            narrow(FeatureType, data["feature"]),  # type: ignore
            description=_get_value(str, data, "description"),
            func=None,
            require_training_set=_get_value(bool, data, "require_training_set"),
        )


@final
@_feature_type_registry.register("Dict")
class Dict(FeatureType):
    def __init__(
        self,
        feature: dict[str, FeatureType],
        description: str | None = None,
        func: Callable[..., Any] | None = None,
        require_training_set: bool | None = None,
    ) -> None:
        """Initializes Dict object.

        Args:
            feature: Definitions of member types.
            description: See FeatureType.__init__.
            func: See FeatureType.__init__.
            require_training_set: See FeatureType.__init__.
        """
        super().__init__("dict", description, func, require_training_set)
        self._feature = feature

    @property
    def feature(self) -> dict[str, FeatureType]:
        return self._feature

    def serialize(self) -> dict[str, SerializableData]:
        """See Serializable.serialize."""
        data = super().serialize()
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
            feature,
            description=_get_value(str, data, "description"),
            func=None,
            require_training_set=_get_value(bool, data, "require_training_set"),
        )


# @dataclass
# class Position(FeatureType):
#    positions: Optional[list] = None
#
#    def __post_init__(self):
#        super().__post_init__()
#        self.cls_name: str = "Position"


@final
@_feature_type_registry.register("Value")
class Value(FeatureType):
    def __init__(
        self,
        max_value: int | float | None = None,
        min_value: int | float | None = None,
        dtype: str | None = None,
        description: str | None = None,
        func: Callable[..., Any] | None = None,
        require_training_set: bool | None = None,
    ) -> None:
        """Initializes Value object.

        Args:
            max_value: The maximum value (inclusive) of values with int/float dtype.
            min_value: The minimum value (inclusive) of values with int/float dtype.
            dtype: See FeatureType.__init__.
            description: See FeatureType.__init__.
            func: See FeatureType.__init__.
            require_training_set: See FeatureType.__init__.
        """
        # Fix inferred types.
        if dtype == "double":
            dtype = "float64"
        elif dtype == "float":
            dtype = "float32"

        super().__init__(dtype, description, func, require_training_set)
        self._max_value = max_value
        self._min_value = min_value

    @property
    def max_value(self) -> int | float | None:
        return self._max_value

    @property
    def min_value(self) -> int | float | None:
        return self._min_value

    def serialize(self) -> dict[str, SerializableData]:
        """See Serializable.serialize."""
        data = super().serialize()
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
            max_value,
            min_value,
            dtype=_get_value(str, data, "dtype"),
            description=_get_value(str, data, "description"),
            func=None,
            require_training_set=_get_value(bool, data, "require_training_set"),
        )
