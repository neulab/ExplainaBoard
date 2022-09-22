"""Tests for explainaboard.serialization.serializers"""

from __future__ import annotations

from dataclasses import dataclass
import unittest

from explainaboard.serialization import common_registry
from explainaboard.serialization.registry import TypeRegistry
from explainaboard.serialization.serializers import PrimitiveSerializer
from explainaboard.serialization.types import (
    Serializable,
    SerializableData,
    SerializableDataclass,
)
from explainaboard.utils.typing_utils import narrow

test_registry = TypeRegistry[Serializable]()


@test_registry.register("Foo")
class Foo(Serializable):
    """Serializable class."""

    def __init__(self, a: int, b: str) -> None:
        self._a = a
        self._b = b

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Foo) and self._a == other._a and self._b == other._b

    def serialize(self) -> dict[str, SerializableData]:
        return {"a": self._a, "b": self._b}

    @classmethod
    def deserialize(cls, data: dict[str, SerializableData]) -> Serializable:
        return cls(narrow(int, data["a"]), narrow(str, data["b"]))


@test_registry.register("Bar")
@dataclass(frozen=True)
class Bar(Serializable):
    """Serializable dataclass."""

    x: int
    y: str

    def serialize(self) -> dict[str, SerializableData]:
        return {"x": self.x, "y": self.y}

    @classmethod
    def deserialize(cls, data: dict[str, SerializableData]) -> Serializable:
        return cls(narrow(int, data["x"]), narrow(str, data["y"]))


@test_registry.register("Baz")
@dataclass(frozen=True)
class Baz(SerializableDataclass):
    """Serializable dataclass with mixin."""

    n: int
    m: str


@test_registry.register("Nested")
@dataclass(frozen=True)
class Nested(Serializable):
    """Nested serializable class."""

    foo: Foo
    bar: Bar
    baz: Baz

    def serialize(self) -> dict[str, SerializableData]:
        return {"foo": self.foo, "bar": self.bar, "baz": self.baz}

    @classmethod
    def deserialize(cls, data: dict[str, SerializableData]) -> Serializable:
        return cls(
            narrow(Foo, data["foo"]), narrow(Bar, data["bar"]), narrow(Baz, data["baz"])
        )


class Unregistered(Serializable):
    """Unregistered serializable class."""

    def serialize(self) -> dict[str, SerializableData]:
        raise NotImplementedError

    @classmethod
    def deserialize(cls, data: dict[str, SerializableData]) -> Serializable:
        raise NotImplementedError


class Unserializable:
    """Unserializable class."""

    pass


@test_registry.register("WithClsName")
class WithClsName(Serializable):
    """Class with "cls_name" attribute."""

    def serialize(self) -> dict[str, SerializableData]:
        return {"cls_name": None}

    @classmethod
    def deserialize(cls, _data: dict[str, SerializableData]) -> Serializable:
        return cls()


class PrimitiveSerializerTest(unittest.TestCase):
    def test_init_with_default_registry(self) -> None:
        s = PrimitiveSerializer()
        self.assertIs(s._registry, common_registry)

    def test_serialize_primitives(self) -> None:
        s = PrimitiveSerializer(test_registry)

        self.assertIs(s.serialize(None), None)
        self.assertIs(s.serialize(True), True)
        self.assertIs(s.serialize(False), False)

        self.assertIsInstance(s.serialize(12345), int)
        self.assertIsInstance(s.serialize(123.5), float)
        self.assertIsInstance(s.serialize("12345"), str)
        self.assertIsInstance(s.serialize([1, 2, 3]), list)
        self.assertIsInstance(s.serialize((1, 2, 3)), tuple)
        self.assertIsInstance(s.serialize({"1": 10, "2": 20}), dict)

        self.assertEqual(s.serialize(12345), 12345)
        self.assertEqual(s.serialize(123.5), 123.5)
        self.assertEqual(s.serialize("12345"), "12345")
        self.assertEqual(s.serialize([1, 2, 3]), [1, 2, 3])
        self.assertEqual(s.serialize((1, 2, 3)), (1, 2, 3))
        self.assertEqual(s.serialize({"1": 10, "2": 20}), {"1": 10, "2": 20})

    def test_serialize_serializables(self) -> None:
        s = PrimitiveSerializer(test_registry)

        foo = Foo(111, "222")
        bar = Bar(333, "444")
        baz = Baz(555, "666")
        nested = Nested(foo, bar, baz)

        foo_dict = {"cls_name": "Foo", "a": 111, "b": "222"}
        bar_dict = {"cls_name": "Bar", "x": 333, "y": "444"}
        baz_dict = {"cls_name": "Baz", "n": 555, "m": "666"}
        nested_dict = {
            "cls_name": "Nested",
            "foo": foo_dict,
            "bar": bar_dict,
            "baz": baz_dict,
        }

        self.assertIsInstance(s.serialize(foo), dict)
        self.assertIsInstance(s.serialize(bar), dict)
        self.assertIsInstance(s.serialize(baz), dict)
        self.assertIsInstance(s.serialize(nested), dict)
        self.assertIsInstance(s.serialize([foo, bar, baz]), list)
        self.assertIsInstance(s.serialize((foo, bar, baz)), tuple)
        self.assertIsInstance(s.serialize({"foo": foo, "bar": bar, "baz": baz}), dict)

        self.assertEqual(s.serialize(foo), foo_dict)
        self.assertEqual(s.serialize(bar), bar_dict)
        self.assertEqual(s.serialize(baz), baz_dict)
        self.assertEqual(s.serialize(nested), nested_dict)
        self.assertEqual(s.serialize([foo, bar, baz]), [foo_dict, bar_dict, baz_dict])
        self.assertEqual(s.serialize((foo, bar, baz)), (foo_dict, bar_dict, baz_dict))
        self.assertEqual(
            s.serialize({"foo": foo, "bar": bar, "baz": baz}),
            {"foo": foo_dict, "bar": bar_dict, "baz": baz_dict},
        )

    def test_serialize_invalid(self) -> None:
        s = PrimitiveSerializer(test_registry)

        with self.assertRaisesRegex(ValueError, r"^No name associated"):
            s.serialize(Unregistered())

        with self.assertRaisesRegex(ValueError, r"^Not a serializable data"):
            s.serialize(Unserializable())  # type: ignore

        with self.assertRaisesRegex(ValueError, r"^Serializable can not contain"):
            s.serialize(WithClsName())

        with self.assertRaisesRegex(ValueError, r"^dict can not contain"):
            s.serialize({"cls_name": None})

    def test_deserialize_primitives(self) -> None:
        s = PrimitiveSerializer(test_registry)

        self.assertIs(s.deserialize(None), None)
        self.assertIs(s.deserialize(True), True)
        self.assertIs(s.deserialize(False), False)

        self.assertIsInstance(s.deserialize(12345), int)
        self.assertIsInstance(s.deserialize(123.5), float)
        self.assertIsInstance(s.deserialize("12345"), str)
        self.assertIsInstance(s.deserialize([1, 2, 3]), list)
        self.assertIsInstance(s.deserialize((1, 2, 3)), tuple)
        self.assertIsInstance(s.deserialize({"1": 10, "2": 20}), dict)

        self.assertEqual(s.deserialize(12345), 12345)
        self.assertEqual(s.deserialize(123.5), 123.5)
        self.assertEqual(s.deserialize("12345"), "12345")
        self.assertEqual(s.deserialize([1, 2, 3]), [1, 2, 3])
        self.assertEqual(s.deserialize((1, 2, 3)), (1, 2, 3))
        self.assertEqual(s.deserialize({"1": 10, "2": 20}), {"1": 10, "2": 20})

    def test_deserialize_serializables(self) -> None:
        s = PrimitiveSerializer(test_registry)

        foo = Foo(111, "222")
        bar = Bar(333, "444")
        baz = Baz(555, "666")
        nested = Nested(foo, bar, baz)

        foo_dict = {"cls_name": "Foo", "a": 111, "b": "222"}
        bar_dict = {"cls_name": "Bar", "x": 333, "y": "444"}
        baz_dict = {"cls_name": "Baz", "n": 555, "m": "666"}
        nested_dict = {
            "cls_name": "Nested",
            "foo": foo_dict,
            "bar": bar_dict,
            "baz": baz_dict,
        }

        self.assertIsInstance(s.deserialize(foo_dict), Foo)
        self.assertIsInstance(s.deserialize(bar_dict), Bar)
        self.assertIsInstance(s.deserialize(baz_dict), Baz)
        self.assertIsInstance(s.deserialize(nested_dict), Nested)
        self.assertIsInstance(s.deserialize([foo_dict, bar_dict, baz_dict]), list)
        self.assertIsInstance(s.deserialize((foo_dict, bar_dict, baz_dict)), tuple)
        self.assertIsInstance(
            s.deserialize({"foo": foo_dict, "bar": bar_dict, "baz": baz_dict}), dict
        )

        self.assertEqual(s.deserialize(foo_dict), foo)
        self.assertEqual(s.deserialize(bar_dict), bar)
        self.assertEqual(s.deserialize(baz_dict), baz)
        self.assertEqual(s.deserialize(nested_dict), nested)
        self.assertEqual(s.deserialize([foo_dict, bar_dict, baz_dict]), [foo, bar, baz])
        self.assertEqual(s.deserialize((foo_dict, bar_dict, baz_dict)), (foo, bar, baz))
        self.assertEqual(
            s.deserialize({"foo": foo_dict, "bar": bar_dict, "baz": baz_dict}),
            {"foo": foo, "bar": bar, "baz": baz},
        )

    def test_deserialize_invalid(self) -> None:
        s = PrimitiveSerializer(test_registry)

        with self.assertRaisesRegex(ValueError, r"^Value of \"cls_name\" must be str"):
            # Ensure that this case should be prevented.
            s.deserialize({"cls_name": None})

        with self.assertRaisesRegex(ValueError, r"^Value of \"cls_name\" must be str"):
            s.deserialize({"cls_name": 123})

        with self.assertRaisesRegex(ValueError, r"^No type associated"):
            s.deserialize({"cls_name": "Unknown"})

        with self.assertRaisesRegex(ValueError, r"^Not a serialized data"):
            s.deserialize(Foo(111, "222"))  # type: ignore
