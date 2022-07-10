from __future__ import annotations

from eaas.endpoint import EndpointConfig

from explainaboard.metrics.metric import MetricConfig

_metric_config_registry = {}


def register_metric_config(cls):
    """
    a register for task specific processors.
    example usage: `@register_processor()`
    """

    _metric_config_registry[cls.__name__] = cls
    if cls.__name__.endswith('Config'):
        _metric_config_registry[cls.__name__[:-6]] = cls
    return cls


def metric_name_to_config_class(name):
    return _metric_config_registry[name]


def metric_name_to_config(
    name: str, source_language: str, target_language: str
) -> MetricConfig:
    if name in _metric_config_registry:
        return _metric_config_registry[name](
            name=name, source_language=source_language, target_language=target_language
        )
    elif name in EndpointConfig().valid_metrics:
        return _metric_config_registry['EaaSMetricConfig'](
            name=name,
            source_language=source_language,
            target_language=target_language,
        )
    else:
        raise ValueError(f'Invalid metric {name}')
