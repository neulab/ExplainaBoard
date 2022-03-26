import os
import pathlib
from unittest import TestCase

from explainaboard import FileType, get_loader, Source, TaskType
from explainaboard.loaders.file_loader import FileLoaderField, TSVFileLoader
from explainaboard.loaders.loader import Loader
from explainaboard.tests.utils import load_file_as_str

artifacts_path = os.path.dirname(pathlib.Path(__file__)) + "/artifacts/"


class BaseLoaderTests(TestCase):
    def test_load_in_memory_tsv(self):
        loader = Loader(
            load_file_as_str(f"{artifacts_path}sys_out1.tsv"),
            Source.in_memory,
            FileType.tsv,
            {
                FileType.tsv: TSVFileLoader(
                    [FileLoaderField(0, "field0", str)], use_idx_as_id=True
                )
            },
        )
        samples = [sample for sample in loader.load()]
        self.assertEqual(len(samples), 10)
        self.assertEqual(set(samples[0].keys()), {"id", "field0"})


class FileLoaderTests(TestCase):
    def test_tsv_validation(self):
        self.assertRaises(
            ValueError,
            lambda: TSVFileLoader(
                [FileLoaderField("0", "field0", str)], use_idx_as_id=True
            ),
        )


class TextClassificationLoader(TestCase):
    def test_load_in_memory_tsv(self):
        loader = get_loader(
            TaskType.text_classification,
            load_file_as_str(f"{artifacts_path}sys_out1.tsv"),
            Source.in_memory,
            FileType.tsv,
        )
        data = loader.load()
        self.assertEqual(len(data), 10)
        self.assertEqual(
            set(data[0].keys()), {"id", "text", "true_label", "predicted_label"}
        )


class QASquadLoader(TestCase):
    def test_load_json(self):
        loader = get_loader(
            TaskType.question_answering_extractive,
            f"{artifacts_path}test-qa-squad.json",
            Source.local_filesystem,
            FileType.json,
        )
        data = loader.load()
        self.assertEqual(len(data), 5)


class SummSquadLoader(TestCase):
    def test_load_json(self):
        loader = get_loader(
            TaskType.summarization,
            f"{artifacts_path}test-summ.tsv",
            Source.local_filesystem,
            FileType.tsv,
        )
        data = loader.load()
        self.assertEqual(len(data), 70)
