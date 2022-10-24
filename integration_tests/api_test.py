from __future__ import annotations

import json
import unittest

import requests

from explainaboard.utils.logging import get_logger


class APITest(unittest.TestCase):
    def test_run(self) -> None:
        end_point_upload_dataset = (
            "https://datalab.nlpedia.ai/api/normal_dataset/read_stat"
        )
        data_info = {
            "dataset_name": "sst2",
            "subset_name": None,
            "version": "Hugging Face",
            "transformation": {"type": "origin"},
        }
        response = requests.post(end_point_upload_dataset, json=data_info)

        message = json.loads(response.text.replace("null", ""))["message"]
        get_logger("test").info(message)
        """
        (1) success
        (2) dataset does not exist
        (3) the dataset does not include the information of _stat
        """
        return_content = json.loads(response.content)
        get_logger("test").info(return_content["content"])
