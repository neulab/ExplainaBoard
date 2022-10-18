"""A package for processors."""

from __future__ import annotations

from explainaboard.processors import processor_factory

get_processor_class = processor_factory.get_processor_class
