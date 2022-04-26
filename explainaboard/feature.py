from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
import re
import sys
from typing import Any, Optional

import numpy as np
import pandas as pd
import pyarrow as pa

from explainaboard import config


def _arrow_to_datasets_dtype(arrow_type: pa.DataType) -> str:
    """
    _arrow_to_datasets_dtype takes a pyarrow.DataType and converts it to a datasets
    string dtype. In effect, `dt == string_to_arrow(_arrow_to_datasets_dtype(dt))`
    """

    if pa.types.is_null(arrow_type):
        return "null"
    elif pa.types.is_boolean(arrow_type):
        return "bool"
    elif pa.types.is_int8(arrow_type):
        return "int8"
    elif pa.types.is_int16(arrow_type):
        return "int16"
    elif pa.types.is_int32(arrow_type):
        return "int32"
    elif pa.types.is_int64(arrow_type):
        return "int64"
    elif pa.types.is_uint8(arrow_type):
        return "uint8"
    elif pa.types.is_uint16(arrow_type):
        return "uint16"
    elif pa.types.is_uint32(arrow_type):
        return "uint32"
    elif pa.types.is_uint64(arrow_type):
        return "uint64"
    elif pa.types.is_float16(arrow_type):
        return "float16"  # pyarrow dtype is "halffloat"
    elif pa.types.is_float32(arrow_type):
        return "float32"  # pyarrow dtype is "float"
    elif pa.types.is_float64(arrow_type):
        return "float64"  # pyarrow dtype is "double"
    elif pa.types.is_timestamp(arrow_type):
        assert isinstance(arrow_type, pa.TimestampType)
        if arrow_type.tz is None:
            return f"timestamp[{arrow_type.unit}]"
        elif arrow_type.tz:
            return f"timestamp[{arrow_type.unit}, tz={arrow_type.tz}]"
        else:
            raise ValueError(f"Unexpected timestamp object {arrow_type}.")
    elif pa.types.is_binary(arrow_type):
        return "binary"
    elif pa.types.is_large_binary(arrow_type):
        return "large_binary"
    elif pa.types.is_string(arrow_type):
        return "string"
    elif pa.types.is_large_string(arrow_type):
        return "large_string"
    else:
        raise ValueError(
            f"Arrow type {arrow_type} does not have a datasets dtype equivalent."
        )


def string_to_arrow(datasets_dtype: str) -> pa.DataType:
    """
    string_to_arrow takes a datasets string dtype and converts it to a pyarrow.DataType.
    In effect, `dt == string_to_arrow(_arrow_to_datasets_dtype(dt))`
    This is necessary because the datasets.Value() primitive type is constructed using a
    string dtype Value(dtype=str)
    But Features.type (via `get_nested_type()` expects to resolve Features into a
    pyarrow Schema, which means that each Value() must be able to resolve into a
    corresponding pyarrow.DataType, which is the purpose of this function.
    """
    timestamp_regex = re.compile(r"^timestamp\[(.*)\]$")
    timestamp_matches = timestamp_regex.search(datasets_dtype)
    if timestamp_matches:
        """
        Example timestamp dtypes:
        timestamp[us]
        timestamp[us, tz=America/New_York]
        """
        timestamp_internals = timestamp_matches.group(1)
        internals_regex = re.compile(r"^(s|ms|us|ns),\s*tz=([a-zA-Z0-9/_+\-:]*)$")
        internals_matches = internals_regex.search(timestamp_internals)
        if timestamp_internals in ["s", "ms", "us", "ns"]:
            return pa.timestamp(timestamp_internals)
        elif internals_matches:
            return pa.timestamp(internals_matches.group(1), internals_matches.group(2))
        else:
            raise ValueError(
                f"""
{datasets_dtype} is not a validly formatted string representation of a pyarrow
timestamp. Examples include timestamp[us] or timestamp[us, tz=America/New_York]
See:
https://arrow.apache.org/docs/python/generated/pyarrow.timestamp.html#pyarrow.timestamp
"""
            )
    elif datasets_dtype not in pa.__dict__:
        if str(datasets_dtype + "_") not in pa.__dict__:
            raise ValueError(
                f"""
Neither {datasets_dtype} nor {datasets_dtype + '_'} seems to be a pyarrow data type.
Please make sure to use a correct data type, see:
https://arrow.apache.org/docs/python/api/datatypes.html#factory-functions
"""
            )
        arrow_data_factory_function_name = str(datasets_dtype + "_")
    else:
        arrow_data_factory_function_name = datasets_dtype

    return pa.__dict__[arrow_data_factory_function_name]()


def _cast_to_python_objects(obj: Any, only_1d_for_numpy: bool) -> tuple[Any, bool]:
    """
    Cast pytorch/tensorflow/pandas objects to python numpy array/lists.
    It works recursively.
    To avoid iterating over possibly long lists, it first checks if the first element
    that is not None has to be casted.
    If the first element needs to be casted, then all the elements of the list will be
    casted, otherwise they'll stay the same.
    This trick allows to cast objects that contain tokenizers outputs without iterating
    over every single token for example.
    Args:
        obj: the object (nested struct) to cast
        only_1d_for_numpy (bool): whether to keep the full multi-dim tensors as
            multi-dim numpy arrays, or convert them to nested lists of 1-dimensional
            numpy arrays. This can be useful to keep only 1-d arrays to instantiate
            Arrow arrays. Indeed Arrow only support converting 1-dimensional array
            values.
    Returns:
        casted_obj: the casted object
        has_changed (bool): True if the object has been changed, False if it is
            identical
    """

    if config.TF_AVAILABLE and "tensorflow" in sys.modules:
        import tensorflow as tf

    if config.TORCH_AVAILABLE and "torch" in sys.modules:
        import torch

    if config.JAX_AVAILABLE and "jax" in sys.modules:
        import jax.numpy as jnp

    if isinstance(obj, np.ndarray):
        if not only_1d_for_numpy or obj.ndim == 1:
            return obj, False
        else:
            return (
                [
                    _cast_to_python_objects(x, only_1d_for_numpy=only_1d_for_numpy)[0]
                    for x in obj
                ],
                True,
            )
    elif (
        config.TORCH_AVAILABLE
        and "torch" in sys.modules
        and isinstance(obj, torch.Tensor)
    ):
        if not only_1d_for_numpy or obj.ndim == 1:
            return obj.detach().cpu().numpy(), True
        else:
            return (
                [
                    _cast_to_python_objects(x, only_1d_for_numpy=only_1d_for_numpy)[0]
                    for x in obj.detach().cpu().numpy()
                ],
                True,
            )
    elif (
        config.TF_AVAILABLE
        and "tensorflow" in sys.modules
        and isinstance(obj, tf.Tensor)
    ):
        if not only_1d_for_numpy or obj.ndim == 1:
            return obj.numpy(), True
        else:
            return (
                [
                    _cast_to_python_objects(x, only_1d_for_numpy=only_1d_for_numpy)[0]
                    for x in obj.numpy()
                ],
                True,
            )
    elif config.JAX_AVAILABLE and "jax" in sys.modules and isinstance(obj, jnp.ndarray):
        if not only_1d_for_numpy or obj.ndim == 1:
            return np.asarray(obj), True
        else:
            return (
                [
                    _cast_to_python_objects(x, only_1d_for_numpy=only_1d_for_numpy)[0]
                    for x in np.asarray(obj)
                ],
                True,
            )
    elif isinstance(obj, pd.Series):
        return obj.values.tolist(), True
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict("list"), True
    elif isinstance(obj, dict):
        output = {}
        has_changed = False
        for k, v in obj.items():
            casted_v, has_changed_v = _cast_to_python_objects(
                v, only_1d_for_numpy=only_1d_for_numpy
            )
            has_changed |= has_changed_v
            output[k] = casted_v
        return output if has_changed else obj, has_changed
    elif isinstance(obj, (list, tuple)):
        if len(obj) > 0:
            for first_elmt in obj:
                if first_elmt is not None:
                    break
            casted_first_elmt, has_changed_first_elmt = _cast_to_python_objects(
                first_elmt, only_1d_for_numpy=only_1d_for_numpy
            )
            if has_changed_first_elmt:
                return (
                    [
                        _cast_to_python_objects(
                            elmt, only_1d_for_numpy=only_1d_for_numpy
                        )[0]
                        for elmt in obj
                    ],
                    True,
                )
            else:
                if isinstance(obj, list):
                    return obj, False
                else:
                    return list(obj), True
        else:
            return obj if isinstance(obj, list) else [], isinstance(obj, tuple)
    else:
        return obj, False


def cast_to_python_objects(obj: Any, only_1d_for_numpy=False) -> Any:
    """
    Cast numpy/pytorch/tensorflow/pandas objects to python lists.
    It works recursively.
    To avoid iterating over possibly long lists, it first checks if the first element
    that is not None has to be casted.
    If the first element needs to be casted, then all the elements of the list will be
    casted, otherwise they'll stay the same.
    This trick allows to cast objects that contain tokenizers outputs without iterating
    over every single token for example.
    Args:
        obj: the object (nested struct) to cast
    Returns:
        casted_obj: the casted object
    """
    return _cast_to_python_objects(obj, only_1d_for_numpy=only_1d_for_numpy)[0]


@dataclass
class BucketInfo:
    """
    The class is used to define a dataclass for bucketing strategy
    Args:
        method: the bucket strategy
        number: the number of buckets to be bucketed
        setting: hyper-paraterms of bucketing
    """

    method: str = "bucket_attribute_specified_bucket_value"
    number: int = 4
    setting: Any = 1  # For different bucket_methods, the settings are diverse


def is_dataclass_dict(obj):
    if (
        not isinstance(obj, dict)
        or '_type' not in obj.keys()
        or obj['_type'] not in FEATURETYPE_REGISTRY.keys()
    ):
        return False
    else:
        return True


def fromdict(obj):
    if not is_dataclass_dict(obj):
        raise TypeError("fromdict() should be called on dict with _type")
    return _fromdict_inner(obj)


def _fromdict_inner(obj):
    # reconstruct the dataclass using the type tag
    if is_dataclass_dict(obj):
        result = {}
        for name, data in obj.items():
            result[name] = _fromdict_inner(data)
        return FEATURETYPE_REGISTRY[obj["_type"]](**result)

    # exactly the same as before (without the tuple clause)
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_fromdict_inner(v) for v in obj)
    elif isinstance(obj, dict):
        return type(obj)(
            (_fromdict_inner(k), _fromdict_inner(v)) for k, v in obj.items()
        )
    else:
        return copy.deepcopy(obj)


@dataclass
class FeatureType:
    dtype: Optional[str] = None
    _type: Optional[str] = None
    description: Optional[str] = None
    is_bucket: bool = False
    bucket_info: Optional[BucketInfo] = None
    require_training_set: bool = False
    id: Optional[str] = None

    @classmethod
    def from_dict(cls, data_dict: dict) -> FeatureType:
        field_names = set(f.name for f in dataclasses.fields(cls))

        return cls(**{k: v for k, v in data_dict.items() if k in field_names})

    def __post_init__(self):
        self._type: str = self.__class__.__name__


@dataclass
class Sequence(FeatureType):
    feature: FeatureType = field(default_factory=FeatureType)

    def __post_init__(self):
        super().__post_init__()
        self.dtype = "list"


@dataclass
class Set(FeatureType):
    feature: dict[str, FeatureType] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        self.dtype = "dict"


@dataclass
class Position(FeatureType):
    positions: Optional[list] = None

    def __post_init__(self):
        super().__post_init__()
        self._type: str = "Position"


@dataclass
class Value(FeatureType):

    max_value: Optional[float | int] = None
    min_value: Optional[float | int] = None

    def __post_init__(self):
        super().__post_init__()
        if self.is_bucket and self.bucket_info is None:
            self.bucket_info = BucketInfo(
                method="bucket_attribute_specified_bucket_value",
                number=4,
                setting=(),
            )
        if self.dtype == "double":  # fix inferred type
            self.dtype = "float64"
        if self.dtype == "float":  # fix inferred type
            self.dtype = "float32"


FEATURETYPE_REGISTRY = {
    "FeatureType": FeatureType,
    "Sequence": Sequence,
    "Set": Set,
    "Position": Position,
    "Value": Value,
}


class Features(dict):
    def get_bucket_features(self, include_training_dependent=True) -> list[str]:
        """
        Get features that would be used for bucketing
        :param include_training_dependent: Include training-set dependent features
        :return:
        a list of features
        """

        bucket_features = []
        dict_res = {}
        for feature_name in self.keys():
            dict_feature = copy.deepcopy(self[feature_name])

            if isinstance(dict_feature, type(Value("float"))):
                dict_res[feature_name] = dict_feature

            elif isinstance(dict_feature, dict):
                for k, v in dict_feature.items():
                    dict_res[k] = v
            else:
                while (
                    not isinstance(dict_feature, dict)
                    and "feature" in dict_feature.__dict__.keys()
                ):
                    dict_feature = dict_feature.feature

                if isinstance(dict_feature, type(Value("float"))):
                    dict_res[feature_name] = dict_feature

                if isinstance(dict_feature, dict):
                    for k, v in dict_feature.items():
                        dict_res[k] = v

            # curr_feature = self[feature_name].copy()
            # while "feature" in self[feature_name].__dict__.keys():
        # print("--++----- leaf features ---++------")
        for feature_name, feature_info in dict_res.items():
            # print(k,v)
            if feature_info.is_bucket:
                if include_training_dependent or not feature_info.require_training_set:
                    bucket_features.append(feature_name)

        return bucket_features

    def get_pre_computed_features(self) -> list:
        """
        Get features that would be used for bucketing
        :return:
        a list of features
        """

        pre_computed_features = []
        dict_res = {}
        for feature_name in self.keys():
            dict_feature = copy.deepcopy(self[feature_name])

            if isinstance(dict_feature, type(Value("float"))):
                dict_res[feature_name] = dict_feature

            elif isinstance(dict_feature, dict):
                for k, v in dict_feature.items():
                    dict_res[k] = v
            else:
                while (
                    not isinstance(dict_feature, dict)
                    and "feature" in dict_feature.__dict__.keys()
                ):
                    dict_feature = dict_feature.feature

                if isinstance(dict_feature, type(Value("float"))):
                    dict_res[feature_name] = dict_feature

                if isinstance(dict_feature, dict):
                    for k, v in dict_feature.items():
                        dict_res[k] = v

            # curr_feature = self[feature_name].copy()
            # while "feature" in self[feature_name].__dict__.keys():
        # print("--++----- leaf features ---++------")
        for feature_name, feature_info in dict_res.items():
            # print(k,v)
            if "require_training_set" not in feature_info.__dict__.keys():
                continue
            if feature_info.require_training_set:
                pre_computed_features.append(feature_name)

        # print("--++----- pre_computed features ---++------")
        # for feature_name in pre_computed_features:
        #     print(feature_name)

        return pre_computed_features
