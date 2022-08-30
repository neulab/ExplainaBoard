from __future__ import annotations

from explainaboard.metrics.metric import MetricConfig
from explainaboard.serialization.registry import TypeRegistry

metric_config_registry = TypeRegistry[MetricConfig]()


def metric_config_from_dict(dikt: dict):
    type = dikt.pop('cls_name')
    config_cls = metric_config_registry.get_type(type)
    return config_cls.deserialize(dikt)
