"""DEPRECATED: do not use this module for new implementations."""

import copy
import dataclasses
from inspect import getsource

from explainaboard.analysis.feature import FeatureType, get_feature_type_serializer
from explainaboard.utils.tokenizer import get_tokenizer_serializer, Tokenizer


def general_to_dict(data):
    """DEPRECATED: do not use this function for new implementations."""
    if isinstance(data, FeatureType):
        return get_feature_type_serializer().serialize(data)
    if isinstance(data, Tokenizer):
        return get_tokenizer_serializer().serialize(data)
    elif hasattr(data, 'to_dict'):
        return general_to_dict(getattr(data, 'to_dict')())
    elif dataclasses.is_dataclass(data):
        return dataclasses.asdict(data, dict_factory=explainaboard_dict_factory)
    elif isinstance(data, dict):
        return {k: general_to_dict(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [general_to_dict(v) for v in data]
    # sanitize functions
    elif callable(data):
        return getsource(data)
    else:
        return copy.deepcopy(data)


def explainaboard_dict_factory(data):
    """DEPRECATED: do not use this function for new implementations.

    This can be used to serialize data through the following command:
    serialized_data = dataclasses.asdict(data, dict_factory=explainaboard_dict_factory)
    """
    return {field: general_to_dict(value) for field, value in data}
