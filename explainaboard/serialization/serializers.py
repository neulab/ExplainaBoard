"""Definition of DictSerializer."""

from __future__ import annotations

from explainaboard.serialization import common_registry
from explainaboard.serialization.registry import TypeRegistry
from explainaboard.serialization.types import (
    PrimitiveData,
    Serializable,
    SerializableData,
)


class PrimitiveSerializer:
    """Serialization from/to primitive types.

    This serializer converts given object to possibly nested objects with only the
    builtin types.

    Serializable objects are converted to a dict with special member "cls_name" to store
    its registered type name. This means that the object should not have an attribute
    with name "cls_name".
    """

    def __init__(self, registry: TypeRegistry[Serializable] | None = None) -> None:
        """Initializes DictSerializer.

        Args:
            registry: TypeRegistry to lookup type information. If None, the common
                registry is used instead.
        """
        self._registry = registry if registry is not None else common_registry

    def serialize(self, data: SerializableData) -> PrimitiveData:
        """Serialize given data to a primitive object.

        Args:
            data: data to be converted.

        Returns:
            Converted data.

        Raises:
            ValueError: Some portion in `data` is not convertible.
        """
        if data is None:
            return data

        if isinstance(data, (bool, int, float, str)):
            return data

        if isinstance(data, (list, tuple)):
            return type(data)(self.serialize(x) for x in data)

        if isinstance(data, dict):
            if "cls_name" in data:
                raise ValueError('dict can not contain the key "cls_name".')
            return {k: self.serialize(v) for k, v in data.items()}

        if isinstance(data, Serializable):
            cls_name = self._registry.get_name(type(data))
            attributes = data.serialize()
            if "cls_name" in attributes:
                raise ValueError('Serializable can not contain the key "cls_name".')
            attributes["cls_name"] = cls_name
            return {k: self.serialize(v) for k, v in attributes.items()}

        raise ValueError(f"Not a serializable data: {type(data).__name__}")

    def deserialize(self, data: PrimitiveData) -> SerializableData:
        """Deserialize given data to the original serializable data.

        Args:
            data: data to be converted.

        Returns:
            Restored data.

        Raises:
            ValueError: Some portion in `data` is not convertible.
        """
        if data is None:
            return data

        if isinstance(data, (bool, int, float, str)):
            return data

        if isinstance(data, (list, tuple)):
            return type(data)(self.deserialize(x) for x in data)

        if isinstance(data, dict):
            # NOTE(odashi):
            # Check the existence of the key without using get() since we couldn't
            # distinguish whether the returned None is the default or the real value.
            if "cls_name" not in data:
                # Restore dict.
                return {k: self.deserialize(v) for k, v in data.items()}

            cls_name = data["cls_name"]

            if isinstance(cls_name, str):
                # Restore Serializable.
                cls = self._registry.get_type(cls_name)
                attributes = {
                    k: self.deserialize(v) for k, v in data.items() if k != "cls_name"
                }
                return cls.deserialize(attributes)

            raise ValueError(
                f'Value of "cls_name" must be str, but got {type(cls_name).__name__}'
            )

        raise ValueError(f"Not a serialized data: {type(data).__name__}")
