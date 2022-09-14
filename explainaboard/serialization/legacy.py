"""DEPRECATED: do not use this module for new implementations."""

import copy
import dataclasses
from inspect import getsource
from typing import Any

from explainaboard.analysis.feature import FeatureType, get_feature_type_serializer
from explainaboard.metrics.metric import MetricConfig
from explainaboard.metrics.registry import get_metric_config_serializer
from explainaboard.utils.tokenizer import get_tokenizer_serializer, Tokenizer


def general_to_dict(data: Any) -> Any:
    """DEPRECATED: do not use this function for new implementations."""
    if isinstance(data, FeatureType):
        return get_feature_type_serializer().serialize(data)
    if isinstance(data, MetricConfig):
        return get_metric_config_serializer().serialize(data)
    if isinstance(data, Tokenizer):
        return get_tokenizer_serializer().serialize(data)
    elif hasattr(data, 'to_dict'):
        return general_to_dict(getattr(data, 'to_dict')())
    elif dataclasses.is_dataclass(data):
        # NOTE(odashi): Simulates dataclasses.asdict(), but processes inner data by
        # general_to_dict to appropriately treat Serializable objects.
        return {
            f.name: general_to_dict(getattr(data, f.name))
            for f in dataclasses.fields(data)
        }
    elif isinstance(data, dict):
        return {k: general_to_dict(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [general_to_dict(v) for v in data]
    # sanitize functions
    elif callable(data):
        return getsource(data)
    else:
        return copy.deepcopy(data)
