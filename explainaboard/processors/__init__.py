"""A package for processors for each test.

When a new processor is implemented, remember to import it here so it gets registered.
"""

from explainaboard.processors import processor_factory

get_processor_class = processor_factory.get_processor_class
