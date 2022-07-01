from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

from explainaboard.analysis.analyses import BucketAnalysis


def is_dataclass_dict(obj):
    """
    this function is used to judge if obj is a dictionary with a key `_type`
    :param obj: a python object with different potential type
    :return: boolean variable
    """
    if (
        not isinstance(obj, dict)
        or '_type' not in obj.keys()
        or obj['_type'] not in FEATURETYPE_REGISTRY.keys()
    ):
        return False
    else:
        return True


def _fromdict_inner(obj):
    """
    This function aim to construct a dataclass based on a potentially nested
    dictionary (obj) recursively
    :param obj: python object
    :return: an object with dataclass
    """
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
    # dtype: declare the data type of a feature, e.g. dict, list, float
    dtype: Optional[str] = None
    # _type: declare the class type of the feature: Sequence, Position
    _type: Optional[str] = None
    # description: descriptive information of a feature
    description: Optional[str] = None
    # is_input: whether the feature is in the input
    is_input: bool = False
    # is_custom: whether this is a custom feature input from outside
    is_custom: bool = False
    # require_training_set: whether calculating this feature
    # relies on the training samples
    require_training_set: bool = False
    id: Optional[str] = None

    @classmethod
    def from_dict(cls, obj: dict) -> FeatureType:
        if not is_dataclass_dict(obj):
            raise TypeError("from_dict() should be called on dict with _type")
        return _fromdict_inner(obj)

    def __post_init__(self):
        self._type: str = self.__class__.__name__


@dataclass
class Sequence(FeatureType):
    feature: FeatureType = field(default_factory=FeatureType)

    def __post_init__(self):
        super().__post_init__()
        self.dtype = "list"


@dataclass
class Dict(FeatureType):
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

    # the maximum value (inclusive) of a feature with the
    # dtype of `float` or `int`
    max_value: Optional[float | int] = None
    # the minimum value (inclusive) of a feature with the
    # dtype of `float` or `int`
    min_value: Optional[float | int] = None

    def __post_init__(self):
        super().__post_init__()
        if self.dtype == "double":  # fix inferred type
            self.dtype = "float64"
        if self.dtype == "float":  # fix inferred type
            self.dtype = "float32"


FEATURETYPE_REGISTRY = {
    "FeatureType": FeatureType,
    "Sequence": Sequence,
    "Dict": Dict,
    "Position": Position,
    "Value": Value,
    "BucketInfo": BucketAnalysis,
}


class Features(dict):

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

        for feature_name, feature_info in dict_res.items():
            if "require_training_set" not in feature_info.__dict__.keys():
                continue
            if feature_info.require_training_set:
                pre_computed_features.append(feature_name)

        return pre_computed_features
