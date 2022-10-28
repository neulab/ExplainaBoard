"""Type definitions for serialization."""

from __future__ import annotations

import abc
import dataclasses
from typing import final, Union

# TODO(odashi):
# Recursive type is supported by only the head of mypy:
# https://github.com/python/mypy/issues/731
# Remove following type-ignore annotations when we got an official release.

# Type of primitive data.
PrimitiveData = Union[  # type: ignore
    None,
    bool,
    int,
    float,
    str,
    list["PrimitiveData"],  # type: ignore
    tuple["PrimitiveData", ...],  # type: ignore
    dict[str, "PrimitiveData"],  # type: ignore
]

# Type of elements in Serializable objects.
SerializableData = Union[  # type: ignore
    None,
    bool,
    int,
    float,
    str,
    list["SerializableData"],  # type: ignore
    tuple["SerializableData", ...],  # type: ignore
    dict[str, "SerializableData"],  # type: ignore
    "Serializable",
]


class Serializable(metaclass=abc.ABCMeta):
    """Interface to represent serializable classes."""

    @abc.abstractmethod
    def serialize(self) -> dict[str, SerializableData]:
        """Returns the dict of inner elements.

        This function must return a dict of shallow references (or copies) to inner
        objects. Any manual expansion should not be applied by itself so that external
        serializers can treat all elements with their manner.

        Returns:
            dict containing the data of self.
        """
        ...

    @classmethod
    @abc.abstractmethod
    def deserialize(cls, data: dict[str, SerializableData]) -> Serializable:
        """Constructs the object corresponding to the given data.

        Args:
            data: dict of elements to reconstruct the object.
                At the point of calling this function, all elements in this argument
                are already reconstructed by the external deserializer.
                Since the deserializer does not have ways to validate member types, this
                function need to check if the given elements have correct types.
                Note that this argument may give elements with wrong types even when the
                serialized data has no errors. For example, several serialization
                formats do not distinguish the difference between list and tuple.

        Returns:
            Reconstructed object.
        """
        ...


class SerializableDataclass(Serializable):
    """Mix-in class of serializable dataclass.

    This class provides a common implementation to serialize/deserialize dataclass.

    Example:
        >>> @dataclass
        >>> MyData(SerializableDataclass):
        >>>     foo: int
        >>>     bar: str

        >>> MyData(111, "222").serialize()
        {"foo": 111, "bar": "222"}

        >>> MyData.deserialize({"foo": 333, "bar": "444"})
        MyData(foo=333, bar="444")
    """

    @final
    def serialize(self) -> dict[str, SerializableData]:
        """See Serializable.serialize."""
        if not dataclasses.is_dataclass(self):
            raise TypeError(f"{self.__class__.__name__} is not a dataclass.")
        return dataclasses.asdict(self)

    @final
    @classmethod
    def deserialize(cls, data: dict[str, SerializableData]) -> Serializable:
        """See Serializable.deserialize."""
        if not dataclasses.is_dataclass(cls):
            raise TypeError(f"{cls.__name__} is not a dataclass.")

        # This function does not process runtime type checking for now.
        field_names = set(field.name for field in dataclasses.fields(cls))
        return cls(**{k: v for k, v in data.items() if k in field_names})
