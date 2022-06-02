from enum import Enum


class TaskType(str, Enum):
    text_classification = "text-classification"
    named_entity_recognition = "named-entity-recognition"
    qa_extractive = "qa-extractive"
    summarization = "summarization"
    machine_translation = "machine-translation"
    text_pair_classification = "text-pair-classification"
    aspect_based_sentiment_classification = "aspect-based-sentiment-classification"
    kg_link_tail_prediction = "kg-link-tail-prediction"
    qa_multiple_choice = "qa-multiple-choice"
    conditional_generation = "conditional-generation"
    word_segmentation = "word-segmentation"
    language_modeling = "language-modeling"
    chunking = "chunking"
    cloze_mutiple_choice = "cloze-multiple-choice"
    cloze_generative = "cloze-generative"
    grammatical_error_correction = "grammatical-error-correction"

    @staticmethod
    def list():
        return list(map(lambda c: c.value, TaskType))


class Source(str, Enum):
    in_memory = "in_memory"  # content has been loaded in memory
    local_filesystem = "local_filesystem"
    s3 = "s3"
    mongodb = "mongodb"


class FileType(str, Enum):
    json = "json"
    tsv = "tsv"
    csv = "csv"
    conll = "conll"  # for tagging task such as named entity recognition
    datalab = "datalab"
    text = "text"

    @staticmethod
    def list():
        return list(map(lambda c: c.value, FileType))
