"""Tests for explainaboard.analysis.feature."""

import unittest

from explainaboard.analysis.feature import (
    Dict,
    get_feature_type_serializer,
    Sequence,
    Value,
)


class SequenceTest(unittest.TestCase):
    def test_members(self) -> None:
        def dummy_fn():
            return 123

        feature = Sequence(
            feature=Value(dtype="string"),
            description="test",
            func=dummy_fn,
            require_training_set=True,
        )
        self.assertEqual(feature.dtype, "list")
        self.assertEqual(feature.description, "test")
        self.assertIs(feature.func, dummy_fn)
        self.assertEqual(feature.require_training_set, True)
        self.assertEqual(feature.feature, Value(dtype="string"))

    def test_serialize(self) -> None:
        serializer = get_feature_type_serializer()
        feature = Sequence(
            feature=Value(dtype="string"),
            description="test",
            require_training_set=True,
        )
        serialized = {
            "cls_name": "Sequence",
            "dtype": "list",
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
        serializer = get_feature_type_serializer()
        feature = Sequence(
            feature=Value(dtype="string"),
            description="test",
            require_training_set=True,
        )
        serialized = {
            "cls_name": "Sequence",
            "dtype": "list",
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
            feature={"foo": Value(dtype="string")},
            description="test",
            func=dummy_fn,
            require_training_set=True,
        )
        self.assertEqual(feature.dtype, "dict")
        self.assertEqual(feature.description, "test")
        self.assertIs(feature.func, dummy_fn)
        self.assertEqual(feature.require_training_set, True)
        self.assertEqual(feature.feature, {"foo": Value(dtype="string")})

    def test_serialize(self) -> None:
        serializer = get_feature_type_serializer()
        feature = Dict(
            feature={"foo": Value(dtype="string")},
            description="test",
            require_training_set=True,
        )
        serialized = {
            "cls_name": "Dict",
            "dtype": "dict",
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
        serializer = get_feature_type_serializer()
        feature = Dict(
            feature={"foo": Value(dtype="string")},
            description="test",
            require_training_set=True,
        )
        serialized = {
            "cls_name": "Dict",
            "dtype": "dict",
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
            dtype="string",
            description="test",
            func=dummy_fn,
            require_training_set=True,
            max_value=123,
            min_value=45,
        )
        self.assertEqual(feature.dtype, "string")
        self.assertEqual(feature.description, "test")
        self.assertIs(feature.func, dummy_fn)
        self.assertEqual(feature.require_training_set, True)
        self.assertEqual(feature.max_value, 123)
        self.assertEqual(feature.min_value, 45)

    def test_serialize(self) -> None:
        serializer = get_feature_type_serializer()
        feature = Value(
            dtype="string",
            description="test",
            require_training_set=True,
            max_value=123,
            min_value=45,
        )
        serialized = {
            "cls_name": "Value",
            "dtype": "string",
            "max_value": 123,
            "min_value": 45,
            "description": "test",
            "require_training_set": True,
        }
        self.assertEqual(serializer.serialize(feature), serialized)

    def test_deserialize(self) -> None:
        serializer = get_feature_type_serializer()
        feature = Value(
            dtype="string",
            description="test",
            require_training_set=True,
            max_value=123,
            min_value=45,
        )
        serialized = {
            "cls_name": "Value",
            "dtype": "string",
            "max_value": 123,
            "min_value": 45,
            "description": "test",
            "require_training_set": True,
        }
        self.assertEqual(serializer.deserialize(serialized), feature)
