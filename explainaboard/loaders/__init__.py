from explainaboard.loaders import (
    aspect_based_sentiment_classification,
    conditional_generation,
    extractive_qa,
    kg_link_tail_prediction,
    loader,
    loader_registry,
    named_entity_recognition,
    qa_multiple_choice,
    text_classification,
    text_pair_classification,
)

get_loader = loader_registry.get_loader
get_loader_custom_dataset = loader_registry.get_loader_custom_dataset

__all__ = [
    'aspect_based_sentiment_classification',
    'conditional_generation',
    'extractive_qa',
    'kg_link_tail_prediction',
    'loader',
    'named_entity_recognition',
    'qa_multiple_choice',
    'text_classification',
    'text_pair_classification',
]
