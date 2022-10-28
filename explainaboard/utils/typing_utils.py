"""Generic functions to manipulate type hints."""

from __future__ import annotations

from collections.abc import Callable, Generator, Iterable
from typing import Any, TypeVar

T = TypeVar("T")


def unwrap(obj: T | None) -> T:
    """Unwrap the ``Optional`` type hint.

    This function takes an object wrapped with the ``Optional``, and returns itself
    if the object is not ``None``. Otherwise this funciton raises ValueError.

    Args:
        obj: The object to unwrap.

    Returns:
        ``obj`` itself.

    Raises:
        ValueError: ``obj`` is None.
    """
    if obj is None:
        raise ValueError("Attempted to unwrap None.")
    return obj


def unwrap_or(obj: T | None, default: T) -> T:
    """Unwrap the ``Optional`` type hint, or return the default value.

    This function takes an object wrapped with the ``Optional``, and returns itself
    if the object is not ``None``. Otherwise this function returns ``default``.

    Args:
        obj: The object to unwrap.
        default: The default value.

    Returns:
        ``obj`` or ``default`` according to the value of ``obj``.
    """
    return obj if obj is not None else default


def unwrap_or_else(obj: T | None, default: Callable[[], T]) -> T:
    """Unwrap the ``Optional`` type hint, or return the default value.

    This function takes an object wrapped with the ``Optional``, and returns itself
    if the object is not ``None``. Otherwise this function returns the return value of
    ``default``. This means that obtaining the default value is deferred until it is
    required.

    Args:
        obj: The object to unwrap.
        default: The function to return the default value.

    Returns:
        ``obj`` or ``default()`` according to the value of ``obj``.
    """
    return obj if obj is not None else default()


def unwrap_generator(obj: Iterable[T] | None) -> Generator[T, None, None]:
    """Unwrap the ``Optional`` ``Iterable``s and provides its generator.

    This function takes an ``Iterable`` object wrapped by the ``Optional``, and provides
    an iterator over the underlying object. If the object is ``None``, this function
    yields nothing and returns immediately.
    If raising ``ValueError`` when ``None`` is perferred, use ``unwrap()`` instead.

    Args:
        obj: The object to unwrap.

    Returns:
        A generator over the underlying object.
    """
    if obj is not None:
        yield from obj


def narrow(subcls: type[T], obj: Any) -> T:
    """Narrow (downcast) an object with a type-safe manner.

    This function does the same type casting with ``typing.cast()``, but additionally
    checks the actual type of the given object. If the type of the given object is not
    castable to the given type, this funtion raises a ``TypeError``.

    Args:
        subcls: The type that ``obj`` is casted to.
        obj: The object to be casted.

    Returns:
        ``obj`` itself

    Raises:
        TypeError: ``obj`` is not an object of ``T``.
    """
    if not isinstance(obj, subcls):
        raise TypeError(
            f"{obj.__class__.__name__} is not a subclass of {subcls.__name__}"
        )

    # NOTE(odashi): typing.cast() does not work with TypeVar.
    # Simply returning the obj is correct because we already narrowed its type
    # by the previous if-statement.
    return obj
