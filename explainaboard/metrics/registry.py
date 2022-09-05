from __future__ import annotations

from explainaboard.metrics.metric import MetricConfig
from explainaboard.serialization.registry import TypeRegistry
from explainaboard.serialization.serializers import PrimitiveSerializer

metric_config_registry = TypeRegistry[MetricConfig]()


def get_metric_config_serializer() -> PrimitiveSerializer:
    """Create a serializer for metric configs.

    Returns:
        A serializer object for MetricConfig classes.
    """
    return PrimitiveSerializer(metric_config_registry)
