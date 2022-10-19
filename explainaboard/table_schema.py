"""A list of table schemas.

Each schema is a list of dictionary, which is used to instruct how to print bucket-level
cases in the frontend table (the number of list denotes the number of table columns)
Currently, the table schema is characterized by:
(1) field_key:str:  this is used to retrieve data from system output file
(2) sort:bool: whether this column is sortable
(3) filter:bool: whether this column is filterable
(4) label:str: the text to be printed of this column in the table head


For some tasks (e.g., extractive_qa_squad), whose system output format involves nested
dict, for example, extractive QA tasks
{
    "title": title,
    "context": context,
    "question": qa["question"],
    "id": qa["id"],
    "true_answers": {
        "answer_start": answer_starts,
        "text": answers,
    },
    "predicted_answer": pred_answer
}


For this case,  the field_key will be with the format "answer = A.B", suggesting a
nested dict, for example supposing we have `system_output` file and `sample_id`, then we
can get "answer"
answer = system_output[A][B]
"""

from __future__ import annotations

from explainaboard import TaskType

table_schemas = {}
"""Text Classification
text | true_label | predicted_label

Text | True Label | Prediction
"""

table_schemas[TaskType.text_classification] = [
    {"field_key": "text", "sort": False, "filter": False, "label": "Text"},
    {"field_key": "true_label", "sort": True, "filter": True, "label": "True Label"},
    {
        "field_key": "predicted_label",
        "sort": True,
        "filter": True,
        "label": "Prediction",
    },
]


table_schemas[TaskType.text_pair_classification] = [
    {"field_key": "text1", "sort": False, "filter": False, "label": "Text1"},
    {"field_key": "text2", "sort": False, "filter": False, "label": "Text2"},
    {"field_key": "true_label", "sort": True, "filter": True, "label": "True Label"},
    {
        "field_key": "predicted_label",
        "sort": True,
        "filter": True,
        "label": "Prediction",
    },
]


# the case will be directly stored in system_metadata
table_schemas[TaskType.named_entity_recognition] = [
    {"field_key": "span", "sort": False, "filter": True, "label": "Entity"},
    {"field_key": "text", "sort": False, "filter": True, "label": "Sentence"},
    {"field_key": "true_label", "sort": True, "filter": True, "label": "True Label"},
    {
        "field_key": "predicted_label",
        "sort": True,
        "filter": True,
        "label": "Prediction",
    },
]

table_schemas[TaskType.summarization] = [
    {"field_key": "source", "sort": False, "filter": True, "label": "Source Document"},
    {
        "field_key": "references",
        "sort": False,
        "filter": True,
        "label": "Gold References",
    },
    {
        "field_key": "hypothesis",
        "sort": False,
        "filter": True,
        "label": "Generated Summary",
    },
]

table_schemas[TaskType.qa_extractive] = [
    {"field_key": "context", "sort": False, "filter": True, "label": "Context"},
    {"field_key": "question", "sort": False, "filter": True, "label": "Question"},
    {
        "field_key": "true_answers.text",
        "sort": False,
        "filter": True,
        "label": "True Answers",
    },
    {
        "field_key": "predicted_answer",
        "sort": False,
        "filter": True,
        "label": "Predicted Answer",
    },
]

table_schemas[TaskType.aspect_based_sentiment_classification] = [
    {"field_key": "aspect", "sort": False, "filter": False, "label": "Aspect"},
    {"field_key": "text", "sort": False, "filter": False, "label": "Text"},
    {"field_key": "true_label", "sort": True, "filter": True, "label": "True Label"},
    {
        "field_key": "predicted_label",
        "sort": True,
        "filter": True,
        "label": "Prediction",
    },
]
