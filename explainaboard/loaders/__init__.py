from explainaboard.loaders import aspect_based_sentiment_classification
from explainaboard.loaders import conditional_generation
from explainaboard.loaders import extractive_qa
from explainaboard.loaders import kg_link_tail_prediction
from explainaboard.loaders import loader
from explainaboard.loaders import named_entity_recognition
from explainaboard.loaders import qa_multiple_choice
from explainaboard.loaders import text_classification
from explainaboard.loaders import text_pair_classification

get_loader = loader.get_loader

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
