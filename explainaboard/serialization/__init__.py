"""Package definition of explainaboard/serialization."""

from __future__ import annotations

from typing import Final

from explainaboard.serialization.registry import TypeRegistry
from explainaboard.serialization.types import Serializable

# Common registry used across the whole library.
common_registry: Final[TypeRegistry[Serializable]] = TypeRegistry[Serializable]()
