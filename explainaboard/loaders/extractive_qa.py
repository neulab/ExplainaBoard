from typing import Dict, Iterable, List
from explainaboard.constants import Source, FileType
from enum import Enum
from .loader import register_loader
from .loader import Loader
import json
import os
from explainaboard.tasks import TaskType


@register_loader(TaskType.question_answering_extractive)
class QAExtractiveLoader(Loader):
    """ """

    def __init__(self, source: Source, file_type: Enum, data: str = None):

        if source is None:
            source = Source.local_filesystem
        if file_type is None:
            file_type = FileType.json

        self._source = source
        self._file_type = file_type
        self._data = data

    def load(self) -> Iterable[Dict]:
        """
        :param path_system_output: the path of system output file with following format:
        text \t label \t predicted_label
        :return: class object
        """
        raw_data = self._load_raw_data_points()

        # if self._file_type == FileType.json:
        #     pred_json = raw_data
        # else:
        #     raise NotImplementedError

        # this will be replaced by introducing dataset
        # path_test_set = os.path.abspath(
        #     os.path.join(os.path.dirname(__file__), "../datasets/squad/testset-en.json")
        # )

        raw_data = self._load_raw_data_points()  # for json files: loads the entire json
        data: List[Dict] = []
        if self._file_type == FileType.json:
            for id, data_info in enumerate(raw_data):
                data.append(
                    {
                        "id": str(id),  # should be string type
                        "context": data_info["context"],
                        "question": data_info["question"],
                        "answers": data_info["answers"],
                        "predicted_answers": data_info["predicted_answers"],
                    }
                )
        elif self._file_type == FileType.datalab:
            for id, data_info in enumerate(raw_data):
                data.append(
                    {
                        "id": str(id),  # should be string type
                        "context": data_info["context"],
                        "question": data_info["question"],
                        "answers": data_info["answers"],
                        "predicted_answers": {"text": data_info["prediction"]},
                    }
                )
        else:
            raise NotImplementedError
        return data

        #
        #
        # key = 0
        # data: List[Dict] = []
        # with open(path_test_set, encoding="utf-8") as f:
        #     squad = json.load(f)
        #     for article in squad["data"]:
        #         title = article.get("title", "")
        #         for paragraph in article["paragraphs"]:
        #             context = paragraph[
        #                 "context"
        #             ]  # do not strip leading blank spaces GH-2585
        #             for qa in paragraph["qas"]:
        #                 answer_starts = [
        #                     answer["answer_start"] for answer in qa["answers"]
        #                 ]
        #                 answers = [answer["text"] for answer in qa["answers"]]
        #
        #                 pred_answer = ""
        #                 if qa["id"] in pred_json:
        #                     pred_answer = pred_json[qa["id"]]
        #
        #                 # Features currently used are "context", "question", and "answers".
        #                 # Others are extracted here for the ease of future expansions.
        #                 data.append(
        #                     {
        #                         "title": title,
        #                         "context": context,
        #                         "question": qa["question"],
        #                         "id": qa["id"],
        #                         "true_answers": {
        #                             "answer_start": answer_starts,
        #                             "text": answers,
        #                         },
        #                         "predicted_answer": pred_answer,
        #                     }
        #                 )
        #                 key += 1
        return data
