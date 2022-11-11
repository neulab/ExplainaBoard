from __future__ import annotations

import os
import unittest

from integration_tests.utils import test_artifacts_path

from explainaboard import FileType, get_processor_class, TaskType
from explainaboard.loaders.loader_factory import get_loader_class
from explainaboard.metrics.metric import Score
from explainaboard.metrics.text_to_sql import SQLExactSetMatchConfig, SQLExecutionConfig


class TextToSQLTest(unittest.TestCase):
    artifact_path = os.path.join(test_artifacts_path, "text_to_sql")
    json_dataset = os.path.join(artifact_path, "questions.json")
    txt_output = os.path.join(artifact_path, "preds.sql")

    def test_load_custom_dataset_json_txt(self):
        loader = get_loader_class(TaskType.text_to_sql)(
            self.json_dataset,
            self.txt_output,
            dataset_file_type=FileType.json,
            output_file_type=FileType.text,
        )
        data = loader.load()
        self.assertEqual(len(data), 4)
        self.assertEqual(
            data[1],
            {
                "id": "1",
                "db_id": "concert_singer",
                "question": "List the countries which have singers who are over 20.",
                "true_sql": "select distinct country from singer where age > 20",
                "predicted_sql": "select distinct country from singer "
                "where age > 25\tconcert_singer",
            },
        )

    def test_process(self):
        metadata = {
            "task_name": TaskType.text_to_sql,
            "metric_names": ["ExactSetMatch", "Execution"],
            "metric_configs": {
                "ExactSetMatch": SQLExactSetMatchConfig(
                    db_dir="https://storage.googleapis.com/inspired-public-data/"
                    "explainaboard/task_data/text_to_sql/database",
                    table_path="https://storage.googleapis.com/inspired-public-data/"
                    "explainaboard/task_data/text_to_sql/tables/tables.json",
                ),
                "Execution": SQLExecutionConfig(
                    db_dir="https://storage.googleapis.com/inspired-public-data/"
                    "explainaboard/task_data/text_to_sql/database",
                    table_path="https://storage.googleapis.com/inspired-public-data/"
                    "explainaboard/task_data/text_to_sql/tables/tables.json",
                ),
            },
        }
        loader = get_loader_class(TaskType.text_to_sql)(
            self.json_dataset,
            self.txt_output,
            dataset_file_type=FileType.json,
            output_file_type=FileType.text,
        )
        data = loader.load()
        processor = get_processor_class(TaskType.text_to_sql)()
        sys_info = processor.process(metadata, data, skip_failed_analyses=True)
        self.assertEqual(len(sys_info.results.analyses), 12)
        self.assertEqual(
            len(sys_info.results.analyses[2].details.bucket_performances), 2
        )
        self.assertEqual(len(sys_info.results.overall), 1)
        self.assertAlmostEqual(
            sys_info.results.overall["example"]["ExactSetMatch"]
            .get_value(Score, "score")
            .value,
            3.0 / 4.0,
        )
        self.assertAlmostEqual(
            sys_info.results.overall["example"]["Execution"]
            .get_value(Score, "score")
            .value,
            2.0 / 4.0,
        )
