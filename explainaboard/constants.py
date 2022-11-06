"""Definition of constants used in the whole library."""
from __future__ import annotations

from enum import Enum


class TaskType(str, Enum):
    """Task types available in this tool."""

    text_classification = "text-classification"
    named_entity_recognition = "named-entity-recognition"
    qa_extractive = "qa-extractive"
    summarization = "summarization"
    machine_translation = "machine-translation"
    text_pair_classification = "text-pair-classification"
    aspect_based_sentiment_classification = "aspect-based-sentiment-classification"
    kg_link_tail_prediction = "kg-link-tail-prediction"
    qa_multiple_choice = "qa-multiple-choice"
    qa_open_domain = "qa-open-domain"
    qa_tat = "qa-tat"
    conditional_generation = "conditional-generation"
    word_segmentation = "word-segmentation"
    language_modeling = "language-modeling"
    chunking = "chunking"
    cloze_mutiple_choice = "cloze-multiple-choice"
    cloze_generative = "cloze-generative"
    grammatical_error_correction = "grammatical-error-correction"
    meta_evaluation_wmt_da = "meta-evaluation-wmt-da"
    tabular_regression = "tabular-regression"
    tabular_classification = "tabular-classification"
    argument_pair_extraction = "argument-pair-extraction"
    ranking_with_context = "ranking-with-context"
    argument_pair_identification = "argument-pair-identification"
    meta_evaluation_nlg = "meta-evaluation-nlg"
    text_to_sql = "text-to-sql"

    @staticmethod
    def list() -> list[str]:
        """Obtains string representations of all values.

        Returns:
            List of all values in str.
        """
        return list(map(lambda c: c.value, TaskType))


class Source(str, Enum):
    """Types of data sources."""

    in_memory = "in_memory"  # content has been loaded in memory
    local_filesystem = "local_filesystem"
    s3 = "s3"
    mongodb = "mongodb"


class FileType(str, Enum):
    """Types of file formats."""

    json = "json"
    tsv = "tsv"
    csv = "csv"
    conll = "conll"  # for tagging task such as named entity recognition
    datalab = "datalab"
    text = "text"

    @staticmethod
    def list() -> list[str]:
        """Obtains string representations of all values.

        Returns:
            List of all values in str.
        """
        return list(map(lambda c: c.value, FileType))
