# when a new processor is implemented, remember to import it here so it gets registered
from explainaboard.processors import (
    aspect_based_sentiment_classification,
    chunking,
    cloze_generative,
    cloze_multiple_choice,
    conditional_generation,
    extractive_qa,
    grammatical_error_correction,
    kg_link_tail_prediction,
    language_modeling,
    machine_translation,
    named_entity_recognition,
    nlg_meta_evaluation,
    qa_multiple_choice,
    qa_open_domain,
    summarization,
    tabular_classification,
    tabular_regression,
    text_classification,
    text_pair_classification,
    word_segmentation,
    qa_table_text_hybrid,
)
from explainaboard.processors.processor_registry import get_processor

__all__ = [
    'aspect_based_sentiment_classification',
    'conditional_generation',
    'extractive_qa',
    'get_processor',
    'kg_link_tail_prediction',
    'language_modeling',
    'machine_translation',
    'named_entity_recognition',
    'qa_multiple_choice',
    'summarization',
    'text_classification',
    'tabular_classification',
    'tabular_regression',
    'text_pair_classification',
    'word_segmentation',
    'chunking',
    'cloze_multiple_choice',
    'cloze_generative',
    'grammatical_error_correction',
    'qa_open_domain',
    'nlg_meta_evaluation',
    'qa_table_text_hybrid',
]
