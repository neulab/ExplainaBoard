from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional
from collections.abc import Iterable
from typing import Tuple, Union
from explainaboard import config
import numpy as np
import pandas as pd
from explainaboard.utils.py_utils import zip_dict
import pyarrow as pa
import re
import copy
import sys


def _arrow_to_datasets_dtype(arrow_type: pa.DataType) -> str:
    """
    _arrow_to_datasets_dtype takes a pyarrow.DataType and converts it to a datasets string dtype.
    In effect, `dt == string_to_arrow(_arrow_to_datasets_dtype(dt))`
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
    This is necessary because the datasets.Value() primitive type is constructed using a string dtype
    Value(dtype=str)
    But Features.type (via `get_nested_type()` expects to resolve Features into a pyarrow Schema,
        which means that each Value() must be able to resolve into a corresponding pyarrow.DataType, which is the
        purpose of this function.
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
                f"{datasets_dtype} is not a validly formatted string representation of a pyarrow timestamp."
                f"Examples include timestamp[us] or timestamp[us, tz=America/New_York]"
                f"See: https://arrow.apache.org/docs/python/generated/pyarrow.timestamp.html#pyarrow.timestamp"
            )
    elif datasets_dtype not in pa.__dict__:
        if str(datasets_dtype + "_") not in pa.__dict__:
            raise ValueError(
                f"Neither {datasets_dtype} nor {datasets_dtype + '_'} seems to be a pyarrow data type. "
                f"Please make sure to use a correct data type, see: "
                f"https://arrow.apache.org/docs/python/api/datatypes.html#factory-functions"
            )
        arrow_data_factory_function_name = str(datasets_dtype + "_")
    else:
        arrow_data_factory_function_name = datasets_dtype

    return pa.__dict__[arrow_data_factory_function_name]()


def _cast_to_python_objects(obj: Any, only_1d_for_numpy: bool) -> Tuple[Any, bool]:
    """
    Cast pytorch/tensorflow/pandas objects to python numpy array/lists.
    It works recursively.
    To avoid iterating over possibly long lists, it first checks if the first element that is not None has to be casted.
    If the first element needs to be casted, then all the elements of the list will be casted, otherwise they'll stay the same.
    This trick allows to cast objects that contain tokenizers outputs without iterating over every single token for example.
    Args:
        obj: the object (nested struct) to cast
        only_1d_for_numpy (bool): whether to keep the full multi-dim tensors as multi-dim numpy arrays, or convert them to
            nested lists of 1-dimensional numpy arrays. This can be useful to keep only 1-d arrays to instantiate Arrow arrays.
            Indeed Arrow only support converting 1-dimensional array values.
    Returns:
        casted_obj: the casted object
        has_changed (bool): True if the object has been changed, False if it is identical
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
            return [
                _cast_to_python_objects(x, only_1d_for_numpy=only_1d_for_numpy)[0]
                for x in obj
            ], True
    elif (
        config.TORCH_AVAILABLE
        and "torch" in sys.modules
        and isinstance(obj, torch.Tensor)
    ):
        if not only_1d_for_numpy or obj.ndim == 1:
            return obj.detach().cpu().numpy(), True
        else:
            return [
                _cast_to_python_objects(x, only_1d_for_numpy=only_1d_for_numpy)[0]
                for x in obj.detach().cpu().numpy()
            ], True
    elif (
        config.TF_AVAILABLE
        and "tensorflow" in sys.modules
        and isinstance(obj, tf.Tensor)
    ):
        if not only_1d_for_numpy or obj.ndim == 1:
            return obj.numpy(), True
        else:
            return [
                _cast_to_python_objects(x, only_1d_for_numpy=only_1d_for_numpy)[0]
                for x in obj.numpy()
            ], True
    elif config.JAX_AVAILABLE and "jax" in sys.modules and isinstance(obj, jnp.ndarray):
        if not only_1d_for_numpy or obj.ndim == 1:
            return np.asarray(obj), True
        else:
            return [
                _cast_to_python_objects(x, only_1d_for_numpy=only_1d_for_numpy)[0]
                for x in np.asarray(obj)
            ], True
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
                return [
                    _cast_to_python_objects(elmt, only_1d_for_numpy=only_1d_for_numpy)[
                        0
                    ]
                    for elmt in obj
                ], True
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
    To avoid iterating over possibly long lists, it first checks if the first element that is not None has to be casted.
    If the first element needs to be casted, then all the elements of the list will be casted, otherwise they'll stay the same.
    This trick allows to cast objects that contain tokenizers outputs without iterating over every single token for example.
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
        _method: the bucket strategy
        _number: the number of buckets to be bucketed
        _settting: hyper-paraterms of bucketing
    """

    _method: str = "bucket_attribute_specified_bucket_value"
    _number: int = 4
    _setting: Any = 1  # For different bucket_methods, the settings are diverse


@dataclass
class ClassLabel:
    """
    (Most part of this class are from huggingface)
    This class is used to define a new dataclass for sample's class label.
    For example, in sentiment classification task, we have sample:
        I love this movie \t  positive
    in this case, "positive" is the class label.
    There are three ways to create a ClassLabel object,
    based on three objects:
        * `num_classes`: create 0 to (num_classes-1) labels
        * `names`: a list of label names
        * `names_file`:a file that consists of the list of labels.

    Args:
        num_classes: int, number of classes. If users adopt this way to create
            an object, then all labels are str(numbers) less than num_classes.
        names: list<str>
        names_file:str, path to a file with names, one per line
    """

    num_classes: int = None
    names: List[str] = None
    description: str = None
    names_file: Optional[str] = None
    id: Optional[str] = None
    is_bucket: bool = False
    require_training_set: bool = False
    is_pre_computed: bool = False
    bucket_info: BucketInfo = None
    # Class Variables
    dtype: ClassVar[str] = "int64"
    _str2int: ClassVar[Dict[str, int]] = None
    _int2str: ClassVar[Dict[int, int]] = None
    _type: str = field(default="ClassLabel", init=False, repr=False)

    def __post_init__(self):
        if self.is_bucket and self.bucket_info is None:
            self.bucket_info = BucketInfo(
                _method="bucket_attribute_discrete_value", _number=4, _setting=1
            )
        if self.names_file is not None and self.names is not None:
            raise ValueError("Please provide either names or names_file but not both")
        # Set self.names
        if self.names is None:
            if self.names_file is not None:
                self.names = self._load_names_from_file(self.names_file)
            elif self.num_classes is not None:
                self.names = [str(i) for i in range(self.num_classes)]
            else:
                raise ValueError(
                    "Please either provide num_classes, names or names_file."
                )
        # Set self.num_classes
        if self.num_classes is None:
            self.num_classes = len(self.names)
        elif self.num_classes != len(self.names):
            raise ValueError(
                "ClassLabel number of names do not match the defined num_classes."
            )
        # Prepare mappings
        self._int2str = [str(name) for name in self.names]
        self._str2int = {name: i for i, name in enumerate(self._int2str)}
        if len(self._int2str) != len(self._str2int):
            raise ValueError(
                "Some label names are duplicated. Each label name should be unique."
            )

    def str2int(self, values: Union[str, Iterable]):
        """Conversion class name string => integer."""
        assert isinstance(values, str) or isinstance(
            values, Iterable
        ), f"Values {values} should be a string or an Iterable (list, numpy array, pytorch, tensorflow tensors)"
        return_list = True
        if isinstance(values, str):
            values = [values]
            return_list = False

        output = []
        for value in values:
            if self._str2int:
                # strip key if not in dict
                if value not in self._str2int:
                    value = str(value).strip()
                output.append(self._str2int[str(value)])
            else:
                # No names provided, try to integerize
                failed_parse = False
                try:
                    output.append(int(value))
                except ValueError:
                    failed_parse = True
                if failed_parse or not 0 <= value < self.num_classes:
                    raise ValueError("Invalid string class label %s" % value)
        return output if return_list else output[0]

    def int2str(self, values: Union[int, Iterable]):
        """Conversion integer => class name string."""
        assert isinstance(values, int) or isinstance(
            values, Iterable
        ), f"Values {values} should be an integer or an Iterable (list, numpy array, pytorch, tensorflow tensors)"
        return_list = True
        if isinstance(values, int):
            values = [values]
            return_list = False

        for v in values:
            if not 0 <= v < self.num_classes:
                raise ValueError("Invalid integer class label %d" % v)

        if self._int2str:
            output = [self._int2str[int(v)] for v in values]
        else:
            # No names provided, return str(values)
            output = [str(v) for v in values]
        return output if return_list else output[0]

    def encode_example(self, example_data):
        if self.num_classes is None:
            raise ValueError(
                "Trying to use ClassLabel feature with undefined number of class. "
                "Please set ClassLabel.names or num_classes."
            )

        # If a string is given, convert to associated integer
        if isinstance(example_data, str):
            example_data = self.str2int(example_data)

        # Allowing -1 to mean no label.
        if not -1 <= example_data < self.num_classes:
            raise ValueError(
                "Class label %d greater than configured num_classes %d"
                % (example_data, self.num_classes)
            )
        return example_data

    @staticmethod
    def _load_names_from_file(names_filepath):
        with open(names_filepath, "r", encoding="utf-8") as f:
            return [
                name.strip() for name in f.read().split("\n") if name.strip()
            ]  # Filter empty names


@dataclass
class Sequence:
    """Construct a list of feature from a single type or a dict of types.
    Mostly here for compatiblity with tfds.
    """

    feature: Any
    length: int = -1
    id: Optional[str] = None
    is_bucket: bool = False
    require_training_set: bool = False
    # Automatically constructed
    dtype: ClassVar[str] = "list"
    pa_type: ClassVar[Any] = None
    _type: str = field(default="Sequence", init=False, repr=False)


@dataclass
class Set:
    feature: dict

    dtype: ClassVar[str] = "dict"
    is_bucket: bool = False
    require_training_set: bool = False
    is_pre_computed: bool = False
    bucket_info: BucketInfo = None
    _type: str = field(default="Set", init=False, repr=False)
    id: Optional[str] = None
    pa_type: ClassVar[Any] = None


@dataclass
class Position:
    positions: list = None
    dtype: ClassVar[str] = Any
    is_bucket: bool = False
    require_training_set: bool = False
    is_pre_computed: bool = False
    bucket_info: BucketInfo = None
    _type: str = field(default="Position", init=False, repr=False)
    id: Optional[str] = None
    pa_type: ClassVar[Any] = None


@dataclass
class Span:
    dtype: ClassVar[str] = Any
    is_bucket: bool = False
    require_training_set: bool = False
    is_pre_computed: bool = False
    bucket_info: BucketInfo = None
    _type: str = field(default="Span", init=False, repr=False)
    id: Optional[str] = None
    pa_type: ClassVar[Any] = None


@dataclass
class Value:
    """
    The Value dtypes are as follows:
    null
    bool
    int8
    int16
    int32
    int64
    uint8
    uint16
    uint32
    uint64
    float16
    float32 (alias float)
    float64 (alias double)
    timestamp[(s|ms|us|ns)]
    timestamp[(s|ms|us|ns), tz=(tzstring)]
    binary
    large_binary
    string
    large_string
    """

    dtype: str  # must be initialized when created
    description: str = None
    is_bucket: bool = False  # don't need to be initialized
    require_training_set: bool = False
    is_pre_computed: bool = False
    bucket_info: BucketInfo = None
    id: Optional[str] = None
    # Automatically constructed
    pa_type: ClassVar[Any] = None
    _type: str = field(default="Value", init=False, repr=False)
    # is_bucket: str = field(default=False, init=False, repr=False)

    def __post_init__(self):
        if self.is_bucket and self.bucket_info is None:
            self.bucket_info = BucketInfo(
                _method="bucket_attribute_specified_bucket_value",
                _number=4,
                _setting=(),
            )
        if self.dtype == "double":  # fix inferred type
            self.dtype = "float64"
        if self.dtype == "float":  # fix inferred type
            self.dtype = "float32"
        self.pa_type = string_to_arrow(self.dtype)

    def __call__(self):
        return self.pa_type

    def encode_example(self, value):
        if pa.types.is_boolean(self.pa_type):
            return bool(value)
        elif pa.types.is_integer(self.pa_type):
            return int(value)
        elif pa.types.is_floating(self.pa_type):
            return float(value)
        else:
            return value


FeatureType = Union[
    dict,
    list,
    tuple,
    ClassLabel,
    Value,
    Sequence,
    Span,
]


def encode_nested_example(schema, obj):
    """Encode a nested example.
    This is used since some features (in particular ClassLabel) have some logic during encoding.
    """
    # Nested structures: we allow dict, list/tuples, sequences
    if isinstance(schema, dict):
        return {
            k: encode_nested_example(sub_schema, sub_obj)
            for k, (sub_schema, sub_obj) in zip_dict(schema, obj)
        }
    elif isinstance(schema, (list, tuple)):
        sub_schema = schema[0]
        return (
            [encode_nested_example(sub_schema, o) for o in obj]
            if obj is not None
            else None
        )
    elif isinstance(schema, Sequence):
        # We allow to reverse list of dict => dict of list for compatiblity with tfds
        if isinstance(schema.feature, dict):
            # dict of list to fill
            list_dict = {}
            if isinstance(obj, (list, tuple)):
                # obj is a list of dict
                for k, dict_tuples in zip_dict(schema.feature, *obj):
                    list_dict[k] = [
                        encode_nested_example(dict_tuples[0], o)
                        for o in dict_tuples[1:]
                    ]
                return list_dict
            else:
                # obj is a single dict
                for k, (sub_schema, sub_objs) in zip_dict(schema.feature, obj):
                    list_dict[k] = [
                        encode_nested_example(sub_schema, o) for o in sub_objs
                    ]
                return list_dict
        # schema.feature is not a dict
        if isinstance(obj, str):  # don't interpret a string as a list
            raise ValueError(
                "Got a string but expected a list instead: '{}'".format(obj)
            )
        return (
            [encode_nested_example(schema.feature, o) for o in obj]
            if obj is not None
            else None
        )
    # Object with special encoding:
    # ClassLabel will convert from string to int, TranslationVariableLanguages does some checks
    elif isinstance(schema, (ClassLabel, Value)):
        return schema.encode_example(obj)
    # Other object should be directly convertible to a native Arrow type (like Translation and Translation)
    return obj


class Features(dict):

    # def __init__(self, dictionary):
    #     print(dictionary)
    #     for k, v in dictionary.items():
    #         setattr(self,k,v)

    def get_bucket_features(self) -> List:
        """
        Get features that would be used for bucketing
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
                bucket_features.append(feature_name)

        # print("--++----- bucket features ---++------")
        # for feature_name in bucket_features:
        #     print(feature_name)

        return bucket_features

    # def get_pre_computed_features(self) -> List:
    #     """
    #     Get features that reply on pre-computed models
    #     :return:
    #     a list of features
    #     """
    #     pre_computed_features = []
    #     for feature_name in self.keys():
    #         if self[feature_name].is_pre_computed:
    #             pre_computed_features.append(feature_name)
    #     return pre_computed_features

    def get_pre_computed_features(self) -> List:
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

    def encode_example(self, example):
        """
        Encode example. (The original version of huggingface is prepared for arrow.)
        Args:
            example (:obj:`dict[str, Any]`): Data in a Dataset row.
        Returns:
            :obj:`dict[str, Any]`
        """
        example = cast_to_python_objects(example)
        return encode_nested_example(self, example)
