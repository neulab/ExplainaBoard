from dataclasses import dataclass, field
from typing import List
from enum import Enum


class TaskType(str, Enum):
    text_classification = "text-classification"
    named_entity_recognition = "named-entity-recognition"
    extractive_qa_squad = "extractive-qa-squad"
    summarization = "summarization"
    machine_translation = "machine-translation"
    text_pair_classification = "text-pair-classification"
    hellaswag = "hellaswag"
    aspect_based_sentiment_classification = "aspect-based-sentiment-classification"
    kg_link_tail_prediction = "kg-link-tail-prediction"
    qa_multiple_choice = "qa-multiple-choice"

    @staticmethod
    def list():
        return list(map(lambda c: c.value, TaskType))


@dataclass
class Task:
    name: str
    description: str = "task description"
    supported: bool = field(default=False)
    supported_metrics: List[str] = field(default_factory=list)
    supported_formats: List[str] = field(default_factory=list)
    supported_datasets: List[str] = field(default_factory=list)


@dataclass
class TaskCategory:
    name: str
    description: str
    tasks: List[Task]


_task_categories: List[TaskCategory] = [
    TaskCategory(
        "conditional-text-generation",
        "data-to-text and text transduction tasks such as translation or summarization",
        [
            Task(
                name=TaskType.machine_translation,
                description="The process of using AI to automatically translate text from one language to another.",
                supported=True,
                supported_metrics=[
                    "bleu",
                    "bart_score_summ",
                    "bart_score_mt",
                    "bart_score_cnn_hypo_ref",
                    "rouge1",
                    "rouge2",
                    "rougeL",
                    "bert_score_f",
                    "bert_score_p",
                    "bert_score_r",
                    "chrf",
                    "comet",
                    "mover_score",
                    "prism",
                ],
                supported_formats=["tsv"],
                supported_datasets=[],
            ),
            Task(
                name=TaskType.summarization,
                description="Summarize long documents into short texts. See more details about the format of upload files: https://github.com/neulab/ExplainaBoard/blob/main/docs/task_summarization.md",
                supported=True,
                supported_metrics=[
                    "bleu",
                    "bart_score_summ",
                    "bart_score_mt",
                    "bart_score_cnn_hypo_ref",
                    "rouge1",
                    "rouge2",
                    "rougeL",
                    "bert_score_f",
                    "bert_score_p",
                    "bert_score_r",
                    "chrf",
                    "comet",
                    "mover_score",
                    "prism",
                ],
                supported_formats=["tsv"],
                supported_datasets=[],
            ),
        ],
    ),
    TaskCategory(
        "text-classification",
        "predicting a class index or boolean value.  ",
        [
            Task(
                name=TaskType.text_classification,
                description="Classify a text into one or multiple predefined categories. See more details about the format of upload files: https://github.com/neulab/ExplainaBoard/blob/main/docs/task_text_classification.md",
                supported=True,
                supported_metrics=["F1score", "Accuracy"],
                supported_formats=["tsv"],
                supported_datasets=[],
            )
        ],
    ),
    TaskCategory(
        "structure-prediction",
        "predicting structural properties of the text, such as syntax",
        [
            Task(
                name=TaskType.named_entity_recognition,
                description="Recognize named entities from a given text. See one example of the uploaded file: https://github.com/neulab/ExplainaBoard/blob/main/data/system_outputs/conll2003/conll2003.elmo",
                supported=True,
                supported_metrics=["f1_score_seqeval"],
                supported_formats=["conll"],
                supported_datasets=[],
            )
        ],
    ),
    TaskCategory(
        "question-answering",
        "question answering tasks",
        [
            Task(
                name=TaskType.extractive_qa_squad,
                description="A task of extracting an answer from a text given a question on the SQuAD dataset. See more details about the format of upload files: https://github.com/neulab/ExplainaBoard/blob/main/docs/task_extractive_qa_squad.md",
                supported=True,
                supported_metrics=["f1_score_qa", "exact_match_qa"],
                supported_formats=["json"],
                supported_datasets=["squad"],
            ),
        ],
    ),
    TaskCategory(
        "span-text-prediction",
        "prediction based on span and text",
        [
            Task(
                name=TaskType.aspect_based_sentiment_classification,
                description="Predict the sentiment of a text based on a specific aspect. See more details about the format of upload files: https://github.com/neulab/ExplainaBoard/blob/main/data/system_outputs/absa/test-aspect.tsv",
                supported=True,
                supported_metrics=["F1score", "Accuracy"],
                supported_formats=["tsv"],
                supported_datasets=[],
            ),
        ],
    ),
    TaskCategory(
        "text-pair-classification",
        "predicting a class of two texts",
        [
            Task(
                name=TaskType.text_pair_classification,
                description="predict the relationship of two texts. See more details about the format of upload files: https://github.com/neulab/ExplainaBoard/blob/main/docs/task_text_pair_classification.md",
                supported=True,
                supported_metrics=["F1score", "Accuracy"],
                supported_formats=["tsv"],
                supported_datasets=[],
            ),
        ],
    ),
    TaskCategory(
        "kg-link-tail-prediction",
        "predicting the tail entity of missing links in knowledge graphs",
        [
            Task(
                name=TaskType.kg_link_tail_prediction,
                description="predicting the tail entity of missing links in knowledge graphs. See more details about the format of upload files: https://github.com/neulab/ExplainaBoard/blob/main/docs/task_kg_link_tail_prediction.md",
                supported=True,
                supported_metrics=["Hits"],
                supported_formats=["json"],
                supported_datasets=[],
            ),
        ],
    ),
    TaskCategory(
        "qa-multiple-choice",
        "Answer a question from multiple options",
        [
            Task(
                name=TaskType.qa_multiple_choice,
                description="Answer a question from multiple options",
                supported=True,
                supported_metrics=["F1score", "Accuracy"],
                supported_formats=["json"],
                supported_datasets=[],
            )
        ],
    ),
]


def get_task_categories():
    """getter for task categories data"""
    return _task_categories
