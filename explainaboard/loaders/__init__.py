from explainaboard.loaders import (
    aspect_based_sentiment_classification,
    conditional_generation,
    extractive_qa,
    file_loader,
    kg_link_tail_prediction,
    language_modeling,
    loader,
    loader_registry,
    qa_multiple_choice,
    sequence_labeling,
    text_classification,
    text_pair_classification,
)

get_datalab_loader = loader_registry.get_datalab_loader
get_custom_dataset_loader = loader_registry.get_custom_dataset_loader
DatalabLoaderOption = file_loader.DatalabLoaderOption

__all__ = [
    'aspect_based_sentiment_classification',
    'conditional_generation',
    'extractive_qa',
    'kg_link_tail_prediction',
    'language_modeling',
    'loader',
    'sequence_labeling',
    'qa_multiple_choice',
    'text_classification',
    'text_pair_classification',
]
