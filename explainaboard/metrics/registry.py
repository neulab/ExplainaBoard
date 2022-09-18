"""A registry for evaluation metrics, so they can be looked up by string."""

from __future__ import annotations

from explainaboard.serialization.registry import TypeRegistry
from explainaboard.serialization.serializers import PrimitiveSerializer
from explainaboard.serialization.types import Serializable

metric_config_registry = TypeRegistry[Serializable]()


def get_metric_config_serializer() -> PrimitiveSerializer:
    """Create a serializer for metric configs.

    Returns:
        A serializer object for MetricConfig classes.
    """
    return PrimitiveSerializer(metric_config_registry)
