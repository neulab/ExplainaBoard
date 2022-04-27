from __future__ import annotations

from dataclasses import dataclass

from explainaboard import TaskType


@dataclass
class Task:
    name: str
    description: str = "task description"


@dataclass
class TaskCategory:
    name: str
    description: str
    tasks: list[Task]


_task_categories: list[TaskCategory] = [
    TaskCategory(
        "conditional-text-generation",
        "data-to-text and text transduction tasks such as translation or summarization",
        [
            Task(
                name=TaskType.machine_translation,
                description=(
                    "The process of using AI to automatically translate text from one "
                    "language to another."
                ),
            ),
            Task(
                name=TaskType.summarization,
                description="""
Summarize long documents into short texts.
See more details about the format of upload files:
https://github.com/neulab/ExplainaBoard/blob/main/docs/task_summarization.md
""",
            ),
            Task(
                name=TaskType.conditional_generation,
                description=(
                    "Generic conditional text generation tasks, e.g., machine "
                    "translation, text summarization"
                ),
            ),
        ],
    ),
    TaskCategory(
        "text-classification",
        "predicting a class index or boolean value.  ",
        [
            Task(
                name=TaskType.text_classification,
                description="""
Classify a text into one or multiple predefined categories.
See more details about the format of upload files:
https://github.com/neulab/ExplainaBoard/blob/main/docs/task_text_classification.md
""",
            )
        ],
    ),
    TaskCategory(
        "structure-prediction",
        "predicting structural properties of the text, such as syntax",
        [
            Task(
                name=TaskType.named_entity_recognition,
                description="""
Recognize named entities from a given text.
See one example of the uploaded file:
https://github.com/neulab/ExplainaBoard/blob/main/data/system_outputs/conll2003/conll2003.elmo
""",
            ),
            Task(
                name=TaskType.word_segmentation,
                description="""
identify word boundaries of some languages (e.g., Chinese).
""",
            ),
        ],
    ),
    TaskCategory(
        "question-answering",
        "question answering tasks",
        [
            Task(
                name=TaskType.question_answering_extractive,
                description="""
A task of extracting an answer from a text given a question.
See more details about the format of upload files:
https://github.com/neulab/ExplainaBoard/blob/main/docs/task_extractive_qa_squad.md
""",
            ),
        ],
    ),
    TaskCategory(
        "span-text-prediction",
        "prediction based on span and text",
        [
            Task(
                name=TaskType.aspect_based_sentiment_classification,
                description="""
Predict the sentiment of a text based on a specific aspect.
See more details about the format of upload files:
https://github.com/neulab/ExplainaBoard/blob/main/data/system_outputs/absa/test-aspect.tsv
""",
            ),
        ],
    ),
    TaskCategory(
        "text-pair-classification",
        "predicting a class of two texts",
        [
            Task(
                name=TaskType.text_pair_classification,
                description="""
predict the relationship of two texts.
See more details about the format of upload files:
https://github.com/neulab/ExplainaBoard/blob/main/docs/task_text_pair_classification.md
""",
            ),
        ],
    ),
    TaskCategory(
        "kg-link-tail-prediction",
        "predicting the tail entity of missing links in knowledge graphs",
        [
            Task(
                name=TaskType.kg_link_tail_prediction,
                description="""
predicting the tail entity of missing links in knowledge graphs.
See more details about the format of upload files:
https://github.com/neulab/ExplainaBoard/blob/main/docs/task_kg_link_tail_prediction.md
""",
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
            )
        ],
    ),
    TaskCategory(
        "language-modeling",
        "Predict the log probability of words in a sequence",
        [
            Task(
                name=TaskType.language_modeling,
                description="Predict the log probability of words in a sequence",
            )
        ],
    ),
]


def get_task_categories():
    """getter for task categories data"""
    return _task_categories
