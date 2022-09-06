import dataclasses
import os
import unittest

from integration_tests.utils import load_file_as_str, test_artifacts_path

from explainaboard.metrics.sql_em_ex import SQLExConfig, SQLEmConfig
from explainaboard import FileType, get_processor, Source, TaskType
from explainaboard.loaders.file_loader import DatalabLoaderOption, FileLoaderMetadata
from explainaboard.loaders.loader_registry import get_loader_class
from explainaboard.utils import cache_api

class TextToSQLTest(unittest.TestCase):
    artifact_path = os.path.join(test_artifacts_path, "text_to_sql")
    json_dataset = os.path.join(artifact_path, "questions.json")
    txt_output = os.path.join(artifact_path, "pred.sql")

    def test_load_custom_dataset_json(self):
        loader = get_loader_class(TaskType.text_to_sql)(
            self.json_dataset,
            self.txt_output,
            dataset_file_type=FileType.json,
            output_file_type=FileType.tsv,
        )
        data = loader.load()
        print(data[0])
        self.assertEqual(len(data), 1034)

    def test_evaluation_metric(self):
        loader = get_loader_class(TaskType.text_to_sql)(
            self.json_dataset,
            self.txt_output,
            dataset_file_type=FileType.json,
            output_file_type=FileType.tsv,
        )
        data = loader.load()
        true = [[d["true_sql"], d["db_id"]] for d in data]
        pred = [[d["predicted_sql"], d["db_id"]] for d in data]
        metric = SQLExConfig(
            name='SQLEx',
            db_dir='https://expressai-xlab.s3.amazonaws.com/large_data/database',

            table_path='https://expressai-xlab.s3.amazonaws.com/large_data/table/tables.json',
            etype='exec'
        ).to_metric()
        result = metric.evaluate(true, pred)
        self.assertAlmostEqual(result.value, 0.793036750483559)

        metric = SQLEmConfig(
            name='SQLEm',
            db_dir='https://expressai-xlab.s3.amazonaws.com/large_data/database',
            table_path='https://expressai-xlab.s3.amazonaws.com/large_data/table/tables.json',
            etype='match'
        ).to_metric()
        result = metric.evaluate(true, pred)
        self.assertAlmostEqual(result.value, 0.7572533849129593)

