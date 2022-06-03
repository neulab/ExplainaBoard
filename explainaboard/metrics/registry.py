from __future__ import annotations

import sys

from eaas.endpoint import EndpointConfig

from explainaboard.metrics.eaas import EaaSMetricConfig
from explainaboard.metrics.metric import MetricConfig


def metric_name_to_config(
    name: str, source_language: str, target_language: str
) -> MetricConfig:
    try:
        metric_module = sys.modules[__name__]
        metric_config = getattr(metric_module, f'{name}Config')
        return metric_config(
            name=name, source_language=source_language, target_language=target_language
        )
    except AttributeError:
        if name in EndpointConfig().valid_metrics:
            return EaaSMetricConfig(
                name=name,
                source_language=source_language,
                target_language=target_language,
            )
        else:
            raise ValueError(f'Invalid metric {name}')
