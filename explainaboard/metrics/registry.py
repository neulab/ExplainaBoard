from __future__ import annotations

import dataclasses
from typing import TypeVar

from explainaboard.metrics.metric import MetricConfig

MetricConfigT = TypeVar("MetricConfigT", bound=MetricConfig)

_metric_config_registry: dict[str, type[MetricConfig]] = {}


def register_metric_config(cls: type[MetricConfigT]) -> type[MetricConfigT]:
    """Decorator function to register a MetricConfig class to the registry.

    Args:
        cls: A MetricConfig subclass.

    Returns:
        `cls` itself. The type information is preserved.
    """
    _metric_config_registry[cls.__name__] = cls
    if cls.__name__.endswith('Config'):
        _metric_config_registry[cls.__name__[:-6]] = cls
    return cls


def get_metric_config_class(name: str) -> type[MetricConfig]:
    """Obtains a MetricConfig class associated to the given name.

    Args:
        name: MetricConfig name.

    Returns:
        MetricConfig subclass associated to `name`.

    Raises:
        ValueError: `name` is not associated.
    """
    config_cls = _metric_config_registry.get(name)
    if config_cls is None:
        raise ValueError(f'Invalid Metric {name}')
    return config_cls


def metric_config_from_dict(dikt: dict):
    type = dikt.pop('cls_name')
    config_cls = get_metric_config_class(type)
    field_names = set(f.name for f in dataclasses.fields(config_cls))
    return config_cls(
        **{k: config_cls.dict_conv(k, v) for k, v in dikt.items() if k in field_names}
    )
