"""Tests for explainaboard.processors.processor_registry"""

from __future__ import annotations

import unittest

from explainaboard import TaskType
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
from explainaboard.processors.processor_registry import get_processor
from explainaboard.processors.qa_multiple_choice import QAMultipleChoiceProcessor
from explainaboard.processors.qa_open_domain import QAOpenDomainProcessor
from explainaboard.processors.summarization import SummarizationProcessor
from explainaboard.processors.tabular_classification import (
    TextClassificationProcessor as TabularClassificationProcessor,
)
from explainaboard.processors.tabular_regression import TabularRegressionProcessor
from explainaboard.processors.text_classification import TextClassificationProcessor
from explainaboard.processors.text_pair_classification import (
    TextPairClassificationProcessor,
)
from explainaboard.processors.word_segmentation import CWSProcessor


class ProcessorRegistryTest(unittest.TestCase):
    def test_get_processor(self) -> None:
        self.assertIsInstance(
            get_processor(TaskType.text_classification.value),
            TextClassificationProcessor,
        )
        self.assertIsInstance(
            get_processor(TaskType.named_entity_recognition.value), NERProcessor
        )
        self.assertIsInstance(
            get_processor(TaskType.qa_extractive.value), QAExtractiveProcessor
        )
        self.assertIsInstance(
            get_processor(TaskType.summarization.value), SummarizationProcessor
        )
        self.assertIsInstance(
            get_processor(TaskType.machine_translation.value),
            MachineTranslationProcessor,
        )
        self.assertIsInstance(
            get_processor(TaskType.text_pair_classification.value),
            TextPairClassificationProcessor,
        )
        self.assertIsInstance(
            get_processor(TaskType.aspect_based_sentiment_classification.value),
            AspectBasedSentimentClassificationProcessor,
        )
        self.assertIsInstance(
            get_processor(TaskType.kg_link_tail_prediction.value),
            KGLinkTailPredictionProcessor,
        )
        self.assertIsInstance(
            get_processor(TaskType.qa_multiple_choice.value), QAMultipleChoiceProcessor
        )
        self.assertIsInstance(
            get_processor(TaskType.qa_open_domain.value), QAOpenDomainProcessor
        )
        self.assertIsInstance(
            get_processor(TaskType.conditional_generation.value),
            ConditionalGenerationProcessor,
        )
        self.assertIsInstance(
            get_processor(TaskType.word_segmentation.value), CWSProcessor
        )
        self.assertIsInstance(
            get_processor(TaskType.language_modeling.value), LanguageModelingProcessor
        )
        self.assertIsInstance(
            get_processor(TaskType.chunking.value),
            ChunkingProcessor,
        )
        self.assertIsInstance(
            get_processor(TaskType.cloze_mutiple_choice.value),
            ClozeMultipleChoiceProcessor,
        )
        self.assertIsInstance(
            get_processor(TaskType.cloze_generative.value), ClozeGenerativeProcessor
        )
        self.assertIsInstance(
            get_processor(TaskType.grammatical_error_correction.value),
            GrammaticalErrorCorrection,
        )
        self.assertIsInstance(
            get_processor(TaskType.nlg_meta_evaluation.value),
            NLGMetaEvaluationProcessor,
        )
        self.assertIsInstance(
            get_processor(TaskType.tabular_regression.value), TabularRegressionProcessor
        )
        self.assertIsInstance(
            get_processor(TaskType.tabular_classification.value),
            TabularClassificationProcessor,
        )

    def test_get_processor_by_str_should_raise_value_error(self) -> None:
        with self.assertRaises(ValueError):
            get_processor("text_classification")
