"""A registry for evaluation metrics, so they can be looked up by string."""

from __future__ import annotations

import dataclasses

from explainaboard.metrics.metric import MetricConfig
from explainaboard.serialization.registry import TypeRegistry

metric_config_registry = TypeRegistry[MetricConfig]()


def metric_config_from_dict(dikt: dict):
    """Create a metric config from a dictionary."""
    type = dikt.pop('cls_name')
    config_cls = metric_config_registry.get_type(type)
    field_names = set(f.name for f in dataclasses.fields(config_cls))
    return config_cls(
        **{k: config_cls.dict_conv(k, v) for k, v in dikt.items() if k in field_names}
    )
