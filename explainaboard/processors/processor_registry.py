"""A registry for processors."""

from __future__ import annotations

from explainaboard import TaskType
from explainaboard.processors.argument_pair_extraction import (
    ArgumentPairExtractionProcessor,
)
from explainaboard.processors.aspect_based_sentiment_classification import (
    AspectBasedSentimentClassificationProcessor,
)
from explainaboard.processors.chunking import ChunkingProcessor
from explainaboard.processors.cloze_generative import ClozeGenerativeProcessor
from explainaboard.processors.cloze_multiple_choice import ClozeMultipleChoiceProcessor
from explainaboard.processors.conditional_generation import (
    ConditionalGenerationProcessor,
)
from explainaboard.processors.extractive_qa import QAExtractiveProcessor
from explainaboard.processors.grammatical_error_correction import (
    GrammaticalErrorCorrection,
)
from explainaboard.processors.kg_link_tail_prediction import (
    KGLinkTailPredictionProcessor,
)
from explainaboard.processors.language_modeling import LanguageModelingProcessor
from explainaboard.processors.machine_translation import MachineTranslationProcessor
from explainaboard.processors.named_entity_recognition import NERProcessor
from explainaboard.processors.nlg_meta_evaluation import NLGMetaEvaluationProcessor
from explainaboard.processors.processor import Processor
from explainaboard.processors.qa_multiple_choice import QAMultipleChoiceProcessor
from explainaboard.processors.qa_open_domain import QAOpenDomainProcessor
from explainaboard.processors.qa_tat import QATatProcessor
from explainaboard.processors.summarization import SummarizationProcessor
from explainaboard.processors.tabular_classification import (
    TabularClassificationProcessor,
)
from explainaboard.processors.tabular_regression import TabularRegressionProcessor
from explainaboard.processors.text_classification import TextClassificationProcessor
from explainaboard.processors.text_pair_classification import (
    TextPairClassificationProcessor,
)
from explainaboard.processors.word_segmentation import CWSProcessor


def get_processor_class(task: TaskType) -> type[Processor]:
    """Returns a Processor class from the given task type.

    Args:
        task: A task type.

    Returns:
        A Processor class associated with the given task type.

    Raises:
        TypeError: if the obtained class is not a subclass of Processor class.
    """
    task_to_procesor = {
        TaskType.text_classification: TextClassificationProcessor,
        TaskType.named_entity_recognition: NERProcessor,
        TaskType.qa_extractive: QAExtractiveProcessor,
        TaskType.summarization: SummarizationProcessor,
        TaskType.machine_translation: MachineTranslationProcessor,
        TaskType.text_pair_classification: TextPairClassificationProcessor,
        TaskType.aspect_based_sentiment_classification: AspectBasedSentimentClassificationProcessor,  # noqa: E501
        TaskType.kg_link_tail_prediction: KGLinkTailPredictionProcessor,
        TaskType.qa_multiple_choice: QAMultipleChoiceProcessor,
        TaskType.qa_open_domain: QAOpenDomainProcessor,
        TaskType.qa_tat: QATatProcessor,
        TaskType.conditional_generation: ConditionalGenerationProcessor,
        TaskType.word_segmentation: CWSProcessor,
        TaskType.language_modeling: LanguageModelingProcessor,
        TaskType.chunking: ChunkingProcessor,
        TaskType.cloze_mutiple_choice: ClozeMultipleChoiceProcessor,
        TaskType.cloze_generative: ClozeGenerativeProcessor,
        TaskType.grammatical_error_correction: GrammaticalErrorCorrection,
        TaskType.nlg_meta_evaluation: NLGMetaEvaluationProcessor,
        TaskType.tabular_regression: TabularRegressionProcessor,
        TaskType.tabular_classification: TabularClassificationProcessor,
        TaskType.argument_pair_extraction: ArgumentPairExtractionProcessor,
    }
    cls = task_to_procesor[task]
    if not issubclass(cls, Processor):
        raise TypeError(f"Obtained class is not a Processor: {cls.__name__}")
    return cls


def get_processor(task: TaskType | str) -> Processor:
    """Return a processor based on the task type.

    TODO: error handling

    Args:
        task: The type of task

    Returns:
        The processor for that task
    """
    cls = get_processor_class(task)
    return cls()
