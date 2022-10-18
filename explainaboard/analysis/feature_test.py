"""Tests for explainaboard.analysis.feature."""

from __future__ import annotations

import unittest

from explainaboard.analysis.feature import DataType, Dict, Sequence, Value
from explainaboard.serialization.serializers import PrimitiveSerializer


class SequenceTest(unittest.TestCase):
    def test_members(self) -> None:
        def dummy_fn():
            return 123

        feature = Sequence(
            feature=Value(dtype=DataType.STRING),
            description="test",
            func=dummy_fn,
            require_training_set=True,
        )
        self.assertEqual(feature.description, "test")
        self.assertIs(feature.func, dummy_fn)
        self.assertEqual(feature.require_training_set, True)
        self.assertEqual(feature.feature, Value(dtype=DataType.STRING))

    def test_serialize(self) -> None:
        serializer = PrimitiveSerializer()
        feature = Sequence(
            feature=Value(dtype=DataType.STRING),
            description="test",
            require_training_set=True,
        )
        serialized = {
            "cls_name": "Sequence",
            "description": "test",
            "require_training_set": True,
            "feature": {
                "cls_name": "Value",
                "dtype": "string",
                "max_value": None,
                "min_value": None,
                "description": None,
                "require_training_set": False,
            },
        }
        self.assertEqual(serializer.serialize(feature), serialized)

    def test_deserialize(self) -> None:
        serializer = PrimitiveSerializer()
        feature = Sequence(
            feature=Value(dtype=DataType.STRING),
            description="test",
            require_training_set=True,
        )
        serialized = {
            "cls_name": "Sequence",
            "description": "test",
            "require_training_set": True,
            "feature": {
                "cls_name": "Value",
                "dtype": "string",
                "max_value": None,
                "min_value": None,
                "description": None,
                "require_training_set": False,
            },
        }
        self.assertEqual(serializer.deserialize(serialized), feature)


class DictTest(unittest.TestCase):
    def test_members(self) -> None:
        def dummy_fn():
            return 123

        feature = Dict(
            feature={"foo": Value(dtype=DataType.STRING)},
            description="test",
            func=dummy_fn,
            require_training_set=True,
        )
        self.assertEqual(feature.description, "test")
        self.assertIs(feature.func, dummy_fn)
        self.assertEqual(feature.require_training_set, True)
        self.assertEqual(feature.feature, {"foo": Value(dtype=DataType.STRING)})

    def test_serialize(self) -> None:
        serializer = PrimitiveSerializer()
        feature = Dict(
            feature={"foo": Value(dtype=DataType.STRING)},
            description="test",
            require_training_set=True,
        )
        serialized = {
            "cls_name": "Dict",
            "description": "test",
            "require_training_set": True,
            "feature": {
                "foo": {
                    "cls_name": "Value",
                    "dtype": "string",
                    "max_value": None,
                    "min_value": None,
                    "description": None,
                    "require_training_set": False,
                },
            },
        }
        self.assertEqual(serializer.serialize(feature), serialized)

    def test_deserialize(self) -> None:
        serializer = PrimitiveSerializer()
        feature = Dict(
            feature={"foo": Value(dtype=DataType.STRING)},
            description="test",
            require_training_set=True,
        )
        serialized = {
            "cls_name": "Dict",
            "description": "test",
            "require_training_set": True,
            "feature": {
                "foo": {
                    "cls_name": "Value",
                    "dtype": "string",
                    "max_value": None,
                    "min_value": None,
                    "description": None,
                    "require_training_set": False,
                },
            },
        }
        self.assertEqual(serializer.deserialize(serialized), feature)


class ValueTest(unittest.TestCase):
    def test_members(self) -> None:
        def dummy_fn():
            return 123

        feature = Value(
            dtype=DataType.INT,
            description="test",
            func=dummy_fn,
            require_training_set=True,
            max_value=123,
            min_value=45,
        )
        self.assertEqual(feature.dtype, DataType.INT)
        self.assertEqual(feature.description, "test")
        self.assertIs(feature.func, dummy_fn)
        self.assertEqual(feature.require_training_set, True)
        self.assertEqual(feature.max_value, 123)
        self.assertEqual(feature.min_value, 45)

    def test_invalid_minmax(self) -> None:
        with self.assertRaisesRegex(ValueError, r"max_value must be greater"):
            Value(dtype=DataType.FLOAT, max_value=1.0, min_value=1.0001)
        with self.assertRaisesRegex(ValueError, r"max_value must be an int"):
            Value(dtype=DataType.INT, max_value=1.0)
        with self.assertRaisesRegex(ValueError, r"min_value must be an int"):
            Value(dtype=DataType.INT, min_value=1.0)
        with self.assertRaisesRegex(ValueError, r"max_value must not be specified"):
            Value(dtype=DataType.STRING, max_value=1.0)
        with self.assertRaisesRegex(ValueError, r"min_value must not be specified"):
            Value(dtype=DataType.STRING, min_value=1.0)

    def test_serialize(self) -> None:
        serializer = PrimitiveSerializer()
        feature = Value(
            dtype=DataType.INT,
            description="test",
            require_training_set=True,
            max_value=123,
            min_value=45,
        )
        serialized = {
            "cls_name": "Value",
            "dtype": "int",
            "max_value": 123,
            "min_value": 45,
            "description": "test",
            "require_training_set": True,
        }
        self.assertEqual(serializer.serialize(feature), serialized)

    def test_deserialize(self) -> None:
        serializer = PrimitiveSerializer()
        feature = Value(
            dtype=DataType.INT,
            description="test",
            require_training_set=True,
            max_value=123,
            min_value=45,
        )
        serialized = {
            "cls_name": "Value",
            "dtype": "int",
            "max_value": 123,
            "min_value": 45,
            "description": "test",
            "require_training_set": True,
        }
        self.assertEqual(serializer.deserialize(serialized), feature)
