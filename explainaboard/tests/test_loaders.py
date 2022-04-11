import os
from unittest import TestCase

from explainaboard import FileType, get_loader, Source, TaskType
from explainaboard.loaders.file_loader import (
    DatalabLoaderOption,
    FileLoaderField,
    TextFileLoader,
    TSVFileLoader,
)
from explainaboard.loaders.loader import Loader
from explainaboard.tests.utils import load_file_as_str, test_artifacts_path


class BaseLoaderTests(TestCase):
    dataset = os.path.join(test_artifacts_path, "text_classification", "dataset.tsv")

    def test_load_in_memory_tsv(self):
        loader = Loader(
            load_file_as_str(self.dataset),
            load_file_as_str(
                os.path.join(test_artifacts_path, "text_classification", "output.txt")
            ),
            Source.in_memory,
            Source.in_memory,
            FileType.tsv,
            FileType.text,
            TSVFileLoader([FileLoaderField(0, "field0", str)], use_idx_as_id=True),
            TextFileLoader("output", str),
        )
        samples = [sample for sample in loader.load()]
        self.assertEqual(len(samples), 10)
        self.assertEqual(set(samples[0].keys()), {"id", "field0", "output"})

    def test_missing_loader(self):
        """raises ValueError because a tsv file loader is not provided by default"""
        self.assertRaises(
            ValueError,
            lambda: Loader(
                self.dataset,
                self.dataset,
                dataset_file_type=FileType.tsv,
                output_file_type=FileType.tsv,
            ),
        )


class TestLoadFromDatalab(TestCase):
    def test_invalid_dataset_name(self):
        loader = get_loader(
            TaskType.text_classification,
            DatalabLoaderOption("invalid_name"),
            output_data="outputdata",
            output_source=Source.in_memory,
            output_file_type=FileType.text,
        )
        self.assertRaises(FileNotFoundError, loader.load)
