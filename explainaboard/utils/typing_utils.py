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


NarrowType = TypeVar("NarrowType")


def narrow(obj: Any, narrow_type: type[NarrowType]) -> NarrowType:
    """returns the object with the narrowed type or raises a TypeError
    (obj: Any, new_type: type[T]) -> T"""
    if isinstance(obj, narrow_type):
        return obj
    else:
        raise TypeError(f"{obj} is expected to be {narrow_type}")
