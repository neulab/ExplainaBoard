import os
from unittest import TestCase

from explainaboard import FileType, get_datalab_loader, Source, TaskType
from explainaboard.loaders import get_custom_dataset_loader
from explainaboard.loaders.file_loader import (
    CoNLLFileLoader,
    DatalabLoaderOption,
    FileLoaderField,
    TextFileLoader,
    TSVFileLoader,
)
from explainaboard.loaders.loader import Loader
from explainaboard.tests.utils import load_file_as_str, test_artifacts_path


class BaseLoaderTests(TestCase):
    dataset = os.path.join(test_artifacts_path, "text_classification", "dataset.tsv")

    def test_add_user_defined_features(self):
        # This test was originally from the KG link prediction task, but it adding it
        # here makes it possible to test that adding user defined features doesn't
        # break anything later in the test suite
        artifact_path = os.path.join(test_artifacts_path, "kg_link_tail_prediction")
        test_data = os.path.join(artifact_path, "data_mini.json")
        dataset_with_custom_feature = os.path.join(
            artifact_path, "with_custom_feature.json"
        )
        loader = get_custom_dataset_loader(  # use defaults
            TaskType.kg_link_tail_prediction,
            test_data,
            dataset_with_custom_feature,
        )
        data = loader.load()
        self.assertEqual(len(data.metadata.custom_features), 1)
        self.assertEqual(len(data), 10)
        self.assertEqual(
            set(data[0].keys()),
            {
                "id",
                "true_head",
                "true_link",
                'true_head_decipher',
                'true_tail_decipher',
                "true_tail",
                "predict",
                "predictions",
                "rel_type",
                "true_rank",
            },
        )

    def test_load_in_memory_tsv(self):
        loader = Loader(
            dataset_data=load_file_as_str(self.dataset),
            output_data=load_file_as_str(
                os.path.join(test_artifacts_path, "text_classification", "output.txt")
            ),
            dataset_source=Source.in_memory,
            output_source=Source.in_memory,
            dataset_file_type=FileType.tsv,
            output_file_type=FileType.text,
            dataset_file_loader=TSVFileLoader(
                [FileLoaderField(0, "field0", str)], use_idx_as_id=True
            ),
            output_file_loader=TextFileLoader("output", str),
        )
        samples = [sample for sample in loader.load()]
        self.assertEqual(len(samples), 10)
        self.assertEqual(set(samples[0].keys()), {"id", "field0", "output"})

    def test_conll_loader(self):
        tabs_path = os.path.join(test_artifacts_path, "ner", "dataset.tsv")
        spaces_path = os.path.join(test_artifacts_path, "ner", "dataset-space.tsv")
        loader_true = CoNLLFileLoader(
            [
                FileLoaderField(0, 'tokens', str),
                FileLoaderField(1, 'true_tags', str),
            ]
        )
        loader_pred = CoNLLFileLoader(
            [
                FileLoaderField(1, 'pred_tags', str),
            ]
        )
        tabs_true = loader_true.load(tabs_path, Source.local_filesystem)
        spaces_true = loader_true.load(spaces_path, Source.local_filesystem)
        self.assertEqual(tabs_true, spaces_true)
        tabs_pred = loader_pred.load(tabs_path, Source.local_filesystem)
        spaces_pred = loader_pred.load(spaces_path, Source.local_filesystem)
        self.assertEqual(tabs_pred, spaces_pred)

    def test_missing_loader(self):
        """raises ValueError because a tsv file loader is not provided by default"""
        self.assertRaises(
            ValueError,
            lambda: Loader(
                dataset_data=self.dataset,
                output_data=self.dataset,
                dataset_file_type=FileType.tsv,
                output_file_type=FileType.tsv,
            ),
        )


class TestLoadFromDatalab(TestCase):
    def test_invalid_dataset_name(self):
        loader = get_datalab_loader(
            task=TaskType.text_classification,
            dataset=DatalabLoaderOption("invalid_name"),
            output_data="outputdata",
            output_source=Source.in_memory,
            output_file_type=FileType.text,
        )
        self.assertRaises(FileNotFoundError, loader.load)
