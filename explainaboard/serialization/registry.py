"""Definition of TypeRegistry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class TypeRegistry(Generic[T]):
    """Registry to store serializable classes.

    This class defines a set of type information used in a serialization format.
    Serializers defined on this registry can store type information to the serialized
    data, while deserializers can in turn restore the type information from the data.

    Specifically, this registry represents a bidirectional mapping between types and
    corresponding strings. Each class can be mapped to only one string.

    Example:
        registry = TypeRegistry[object]()

        # Register existing types
        registry.register("myint")(int)

        # Register a class with decorator
        @registry.register("foo")
        class Foo:
            ...

        # Obtain types
        assert registry.get_type("myint") is int
        assert registry.get_type("foo") is Foo

        # Obtain names
        assert registry.get_name(int) == "myint"
        assert registry.get_name(Foo) == "foo"
    """

    def __init__(self) -> None:
        """Initializes the registry."""
        self._str_to_cls: dict[str, type] = {}
        self._cls_to_str: dict[type, str] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        """Makes a decorator to register a type with the given name.

        Args:
            name: Name of the type in this registry.

        Returns:
            A decorator to register a type.
        """

        def wrapper(cls: type[T]) -> type[T]:
            """Decorator to register a type with the bound name.

            Args:
                cls: Type to be registered.

            Returns:
                `cls` itself.

            Raises:
                ValueError: Either:
                    * `name` is already used in the registry.
                    * `cls` is already registered.
            """
            if name in self._str_to_cls:
                raise ValueError(f'Name "{name}" is already used in the registry.')
            if cls in self._cls_to_str:
                raise ValueError(f'Type "{cls.__name__}" is already registered.')
            self._str_to_cls[name] = cls
            self._cls_to_str[cls] = name
            return cls

        return wrapper

    def get_type(self, name: str) -> type[T]:
        """Obtains the corresponding type of the given name.

        Args:
            name: Name to lookup in this registry.

        Returns:
            The type associated to `name`.

        Raises:
            ValueError: No type associated to `name`.
        """
        cls = self._str_to_cls.get(name)
        if cls is None:
            raise ValueError(f'No type associated to the name "{name}".')
        return cls

    def get_name(self, cls: type[T]) -> str:
        """Obtains the associated name of the given type in this registry.

        Args:
            cls: Type to lookup in this registry.

        Returns:
            The name associated to `cls`.

        Raises:
            ValueError: No name associated to `cls`.
        """
        name = self._cls_to_str.get(cls)
        if name is None:
            raise ValueError(f'No name associated to the type "{cls.__name__}".')
        return name
