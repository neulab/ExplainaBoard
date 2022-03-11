from unittest import TestCase
from explainaboard import TaskType, FileType, Source, get_loader
from explainaboard.tests.utils import load_file_as_str
from explainaboard.loaders.loader import Loader
import pathlib
import os

artifacts_path = os.path.dirname(pathlib.Path(__file__)) + "/artifacts/"


class BaseLoaderTests(TestCase):
    def test_load_in_memory_tsv(self):
        loader = Loader(
            Source.in_memory,
            FileType.tsv,
            load_file_as_str(f"{artifacts_path}sys_out1.tsv"),
        )
        samples = [sample for sample in loader._load_raw_data_points()]
        self.assertEqual(len(samples), 10)
        self.assertEqual(len(samples[0]), 3)


class TextClassificationLoader(TestCase):
    def test_load_in_memory_tsv(self):
        loader = get_loader(
            TaskType.text_classification,
            Source.in_memory,
            FileType.tsv,
            load_file_as_str(f"{artifacts_path}sys_out1.tsv"),
        )
        data = loader.load()
        self.assertEqual(len(data), 10)
        self.assertListEqual(
            list(data[0].keys()), ["id", "text", "true_label", "predicted_label"]
        )


class QASquadLoader(TestCase):
    def test_load_json(self):
        loader = get_loader(
            TaskType.question_answering_extractive,
            Source.local_filesystem,
            FileType.json,
            f"{artifacts_path}test-qa-squad.json",
        )
        data = loader.load()
        # print(data[0].keys())
        # print(len(data))
        self.assertEqual(len(data), 5)


class SummSquadLoader(TestCase):
    def test_load_json(self):
        loader = get_loader(
            TaskType.summarization,
            Source.local_filesystem,
            FileType.tsv,
            f"{artifacts_path}test-summ.tsv",
        )
        data = loader.load()
        # print(data[0].keys())
        # print(len(data))
        self.assertEqual(len(data), 70)
