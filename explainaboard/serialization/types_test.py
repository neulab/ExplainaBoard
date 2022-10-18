"""Tests for explainaboard.serialization.types."""

from __future__ import annotations

import dataclasses
import unittest

from explainaboard.serialization.types import SerializableDataclass


@dataclasses.dataclass
class MyData(SerializableDataclass):
    """SerializableDataclass for this test."""

    foo: int
    bar: str


class WithoutDecorator(SerializableDataclass):
    """SerializableDataclass without decorator."""

    pass


class SerializableDataclassTest(unittest.TestCase):
    def test_serialize(self) -> None:
        self.assertEqual(MyData(111, "222").serialize(), {"foo": 111, "bar": "222"})

    def test_serialize_without_decorator(self) -> None:
        with self.assertRaisesRegex(TypeError, r"is not a dataclass"):
            WithoutDecorator().serialize()

    def test_deserialize(self) -> None:
        self.assertEqual(
            MyData.deserialize({"foo": 333, "bar": "444"}), MyData(333, "444")
        )

    def test_deserialize_without_decorator(self) -> None:
        with self.assertRaisesRegex(TypeError, r"is not a dataclass"):
            WithoutDecorator.deserialize({})

    def test_deserialize_excessive(self) -> None:
        # Unrecognized members are ignored.
        self.assertEqual(
            MyData.deserialize({"foo": 555, "bar": "666", "baz": 777}),
            MyData(555, "666"),
        )

    def test_deserialize_deficient(self) -> None:
        with self.assertRaisesRegex(TypeError, r"positional argument: 'bar'"):
            MyData.deserialize({"foo": 888})
