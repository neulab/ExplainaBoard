"""Tests for explainaboard.serialization.registry"""

from __future__ import annotations

import unittest

from explainaboard.serialization.registry import TypeRegistry


class TypeRegistryTest(unittest.TestCase):
    def test_register_decorator(self) -> None:
        test_registry = TypeRegistry[object]()

        @test_registry.register("foo")
        class Bar:
            pass

        self.assertIs(test_registry.get_type("foo"), Bar)
        self.assertEqual(test_registry.get_name(Bar), "foo")

    def test_register_function(self) -> None:
        test_registry = TypeRegistry[object]()

        class Qux:
            pass

        test_registry.register("baz")(Qux)

        self.assertIs(test_registry.get_type("baz"), Qux)
        self.assertEqual(test_registry.get_name(Qux), "baz")

    def test_register_builtin(self) -> None:
        registry = TypeRegistry[object]()
        registry.register("number")(int)
        self.assertIs(registry.get_type("number"), int)
        self.assertEqual(registry.get_name(int), "number")

    def test_invalid_type(self) -> None:
        registry = TypeRegistry[object]()
        with self.assertRaisesRegex(ValueError, r"^No name associated"):
            registry.get_name(int)

    def test_invalid_name(self) -> None:
        registry = TypeRegistry[object]()
        with self.assertRaisesRegex(ValueError, r"^No type associated"):
            registry.get_type("fail")
