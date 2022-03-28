# when a new processor is implemented, remember to import it here so it gets registered
from explainaboard.processors import (
    aspect_based_sentiment_classification,
    conditional_generation,
    extractive_qa,
    kg_link_tail_prediction,
    machine_translation,
    named_entity_recognition,
    qa_multiple_choice,
    summarization,
    text_classification,
    text_pair_classification,
)
from explainaboard.processors.processor_registry import get_processor

__all__ = [
    'aspect_based_sentiment_classification',
    'conditional_generation',
    'extractive_qa',
    'get_processor',
    'kg_link_tail_prediction',
    'machine_translation',
    'named_entity_recognition',
    'qa_multiple_choice',
    'summarization',
    'text_classification',
    'text_pair_classification',
]
