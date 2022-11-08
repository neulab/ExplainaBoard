"""A registry for Loader classes to look them up by class name."""
from __future__ import annotations

from explainaboard import TaskType
from explainaboard.loaders.argument_pair_extraction import ArgumentPairExtractionLoader
from explainaboard.loaders.aspect_based_sentiment_classification import (
    AspectBasedSentimentClassificationLoader,
)
from explainaboard.loaders.cloze_generative import ClozeGenerativeLoader
from explainaboard.loaders.cloze_multiple_choice import ClozeMultipleChoiceLoader
from explainaboard.loaders.conditional_generation import (
    ConditionalGenerationLoader,
    MachineTranslationLoader,
    SummarizationLoader,
)
from explainaboard.loaders.grammatical_error_correction import (
    GrammaticalErrorCorrectionLoader,
)
from explainaboard.loaders.kg_link_tail_prediction import KgLinkTailPredictionLoader
from explainaboard.loaders.language_modeling import LanguageModelingLoader
from explainaboard.loaders.loader import Loader
from explainaboard.loaders.meta_evaluation_nlg import MetaEvaluationNLGLoader
from explainaboard.loaders.meta_evaluation_wmt_da import MetaEvaluationWMTDALoader
from explainaboard.loaders.qa_extractive import QAExtractiveLoader
from explainaboard.loaders.qa_multiple_choice import QAMultipleChoiceLoader
from explainaboard.loaders.qa_open_domain import QAOpenDomainLoader
from explainaboard.loaders.qa_tat import QATatLoader
from explainaboard.loaders.ranking import RankingwithContextLoader
from explainaboard.loaders.sequence_labeling import SeqLabLoader
from explainaboard.loaders.tabular_classification import TabularClassificationLoader
from explainaboard.loaders.tabular_regression import TabularRegressionLoader
from explainaboard.loaders.text_classification import TextClassificationLoader
from explainaboard.loaders.text_pair_classification import TextPairClassificationLoader
from explainaboard.loaders.text_to_sql import TextToSQLLoader

_LOADERS: dict[TaskType, type[Loader]] = {
    TaskType.argument_pair_extraction: ArgumentPairExtractionLoader,
    TaskType.aspect_based_sentiment_classification: (
        AspectBasedSentimentClassificationLoader
    ),
    TaskType.chunking: SeqLabLoader,
    TaskType.cloze_generative: ClozeGenerativeLoader,
    TaskType.cloze_mutiple_choice: ClozeMultipleChoiceLoader,
    TaskType.conditional_generation: ConditionalGenerationLoader,
    TaskType.grammatical_error_correction: GrammaticalErrorCorrectionLoader,
    TaskType.kg_link_tail_prediction: KgLinkTailPredictionLoader,
    TaskType.language_modeling: LanguageModelingLoader,
    TaskType.machine_translation: MachineTranslationLoader,
    TaskType.named_entity_recognition: SeqLabLoader,
    TaskType.meta_evaluation_wmt_da: MetaEvaluationWMTDALoader,
    TaskType.qa_extractive: QAExtractiveLoader,
    TaskType.qa_multiple_choice: QAMultipleChoiceLoader,
    TaskType.qa_open_domain: QAOpenDomainLoader,
    TaskType.qa_tat: QATatLoader,
    TaskType.summarization: SummarizationLoader,
    TaskType.tabular_classification: TabularClassificationLoader,
    TaskType.tabular_regression: TabularRegressionLoader,
    TaskType.text_classification: TextClassificationLoader,
    TaskType.text_pair_classification: TextPairClassificationLoader,
    TaskType.word_segmentation: SeqLabLoader,
    TaskType.meta_evaluation_nlg: MetaEvaluationNLGLoader,
    TaskType.argument_pair_identification: RankingwithContextLoader,
    TaskType.ranking_with_context: RankingwithContextLoader,
    TaskType.text_to_sql: TextToSQLLoader,
}


def get_loader_class(task: TaskType) -> type[Loader]:
    """Obtains the loader class for the specified task type.

    Args:
        task: Task type.

    Returns:
        The Loader class associated to `task`.
    """
    return _LOADERS[task]
