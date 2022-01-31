from dataclasses import dataclass, field
from typing import List
from enum import Enum


@dataclass
class Task:
    """
    TODO: add supported_file_types
    """
    name: str
    supported: bool = field(default=False)
    supported_metrics: List[str] = field(default_factory=list)


@dataclass
class TaskCategory:
    name: str
    description: str
    tasks: List[Task]


class TaskType(str, Enum):
    text_classification = "text-classification"
    named_entity_recognition = "named-entity-recognition"
    extractive_qa = "extractive-qa"
    summarization = "summarization"
    text_pair_classification = "text-pair-classification"
    hellaswag = "hellaswag"
    aspect_based_sentiment_classification = "aspect-based-sentiment-classification"

    @staticmethod
    def list():
        return list(map(lambda c: c.value, TaskType))


@dataclass
class Task:
    """
    TODO: add supported_file_types
    """
    name: str
    supported: bool = field(default=False)
    supported_metrics: List[str] = field(default_factory=list)


@dataclass
class TaskCategory:
    name: str
    description: str
    tasks: List[Task]


_task_categories: List[TaskCategory] = [
    TaskCategory("conditional-text-generation",
                 "data-to-text and text transduction tasks such as translation or summarization",
                 [
                     Task("machine-translation"),
                     Task("sentence-splitting-fusion"),
                     Task(TaskType.summarization, True, [
                          "bleu", "rouge1", "rouge2", "rougel"])
                 ]),
    TaskCategory("text-classification", "predicting a class index or boolean value",
                 [Task(TaskType.text_classification, True, ["F1score", "Accuracy"])]),
    TaskCategory("structure-prediction", "predicting structural properties of the text, such as syntax",
                 [Task(TaskType.named_entity_recognition, True, ["f1_score_seqeval"])]),
    TaskCategory("question-answering", "question answering tasks",
                 [Task(TaskType.extractive_qa, True, ["f1_score_qa", "exact_match_qa"])]),
    TaskCategory("span-text-prediction", "prediction based on span and text",
                 [Task(TaskType.aspect_based_sentiment_classification, True, ["F1score", "Accuracy"])]),
    TaskCategory("text-pair-classification", "predicting a class of two texts",
                 [Task(TaskType.text_pair_classification, True, ["F1score", "Accuracy"])]),
]


def get_task_categories():
    """getter for task categories data"""
    return _task_categories


def get_task_categories():
    """getter for task categories data"""
    return _task_categories
