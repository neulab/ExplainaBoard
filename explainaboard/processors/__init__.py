# when a new processor is implemented, remember to import it here so it gets registered
from explainaboard.processors import aspect_based_sentiment_classification
from explainaboard.processors import conditional_generation
from explainaboard.processors import extractive_qa
from explainaboard.processors import kg_link_tail_prediction
from explainaboard.processors import machine_translation
from explainaboard.processors import named_entity_recognition
from explainaboard.processors import qa_multiple_choice
from explainaboard.processors import summarization
from explainaboard.processors import text_classification
from explainaboard.processors import text_pair_classification
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
