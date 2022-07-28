from explainaboard.loaders import (
    aspect_based_sentiment_classification,
    cloze_generative,
    cloze_multiple_choice,
    conditional_generation,
    extractive_qa,
    file_loader,
    grammatical_error_correction,
    kg_link_tail_prediction,
    language_modeling,
    loader,
    loader_registry,
    qa_multiple_choice,
    sequence_labeling,
    tabular_classification,
    tabular_regression,
    text_classification,
    text_pair_classification,
)

get_loader_class = loader_registry.get_loader_class
DatalabLoaderOption = file_loader.DatalabLoaderOption

__all__ = [
    'aspect_based_sentiment_classification',
    'cloze_multiple_choice',
    'conditional_generation',
    'extractive_qa',
    'kg_link_tail_prediction',
    'language_modeling',
    'loader',
    'sequence_labeling',
    'qa_multiple_choice',
    'tabular_classification',
    'tabular_regression',
    'text_classification',
    'text_pair_classification',
    'cloze_generative',
    'grammatical_error_correction',
]
