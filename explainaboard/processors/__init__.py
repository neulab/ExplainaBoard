"""A package for processors."""

from explainaboard.processors import processor_factory

from explainaboard.processors.processor_registry import get_processor
get_processor_class = processor_factory.get_processor_class
 
