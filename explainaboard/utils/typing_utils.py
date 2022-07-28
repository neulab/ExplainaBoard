"""Generic functions to manipulate type hints."""

from __future__ import annotations

from collections.abc import Generator, Iterable
from typing import Any, Optional, TypeVar

T = TypeVar('T')


def unwrap(obj: Optional[T]) -> T:
    '''Unwrap the ``Optional`` type hint.

    This function takes an object wrapped with the ``Optional``, and returns itself
    if the object is not ``None``. Otherwise this funciton raises ValueError.

    :param obj: The object to unwrap.
    :type obj: ``Optional[T]``
    :return: ``obj`` itself.
    :rtype: The underlying type ``T``.
    :raises ValueError: ``obj`` is None.
    '''
    if obj is None:
        raise ValueError('Attempted to unwrap None.')
    return obj


def unwrap_generator(obj: Optional[Iterable[T]]) -> Generator[T, None, None]:
    '''Unwrap the ``Optional`` ``Iterable``s and provides its generator.

    This function takes an ``Iterable`` object wrapped by the ``Optional``, and provides
    an iterator over the underlying object. If the object is ``None``, this function
    yields nothing and returns immediately.
    If raising ``ValueError`` when ``None`` is perferred, use ``unwrap()`` instead.

    :param obj: The object to unwrap.
    :type obj: ``Optional[Iterable[T]]``
    :return: A generator over the underlying object.
    :rtype: ``Generator[T, None, None]``
    '''
    if obj is not None:
        yield from obj


def narrow(subcls: type[T], obj: Any) -> T:
    """Narrow (downcast) an object with a type-safe manner.

    This function does the same type casting with ``typing.cast()``, but additionally
    checks the actual type of the given object. If the type of the given object is not
    castable to the given type, this funtion raises a ``TypeError``.

    :param subcls: The type that ``obj`` is casted to.
    :type subcls: ``type[T]``
    :param obj: The object to be casted.
    :type obj: ``Any``
    :return: ``obj`` itself
    :rtype: ``T``
    :raises TypeError: ``obj`` is not an object of ``T``.
    """
    if not isinstance(obj, subcls):
        raise TypeError(
            f"{obj.__class__.__name__} is not a subclass of {subcls.__name__}"
        )

    # NOTE(odashi): typing.cast() does not work with TypeVar.
    # Simply returning the obj is correct because we already narrowed its type
    # by the previous if-statement.
    return obj
