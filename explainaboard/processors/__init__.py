# flake8: noqa
# when a new processor is implemented, remember to import it here so it gets registered
from . import text_classification
from . import named_entity_recognition
from . import extractive_qa
from . import conditional_generation

from . import text_pair_classification
from . import hellaswag

from . import aspect_based_sentiment_classification

from . import kg_link_tail_prediction
from . import qa_multiple_choice
from .processor_registry import get_processor
