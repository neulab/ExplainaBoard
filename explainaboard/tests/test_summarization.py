import os
import unittest

import numpy as np

from explainaboard import FileType, get_processor, Source, TaskType
from explainaboard.loaders import get_loader_class
from explainaboard.loaders.file_loader import DatalabLoaderOption
from explainaboard.metrics.external_eval import ExternalEvalConfig
from explainaboard.tests.utils import OPTIONAL_TEST_SUITES, test_artifacts_path
from explainaboard.utils import cache_api


class TestSummarization(unittest.TestCase):
    artifact_path = os.path.join(test_artifacts_path, "summarization")
    tsv_dataset = os.path.join(artifact_path, "dataset.tsv")
    txt_output = os.path.join(artifact_path, "output.txt")

    def test_load_tsv(self):
        loader = get_loader_class(TaskType.summarization)(
            self.tsv_dataset,
            self.txt_output,
            Source.local_filesystem,
            Source.local_filesystem,
            FileType.tsv,
            FileType.text,
        )
        data = loader.load()
        self.assertEqual(len(data), 3)
        sample = data[0]
        self.assertTrue(sample["source"].startswith("washington"))
        self.assertTrue(sample["reference"].startswith("in an"))
        self.assertTrue(sample["hypothesis"].startswith("washington"))

    def test_generate_system_analysis(self):
        loader = get_loader_class(TaskType.summarization)(
            self.tsv_dataset,
            self.txt_output,
            Source.local_filesystem,
            Source.local_filesystem,
            FileType.tsv,
            FileType.text,
        )
        data = loader.load()

        metadata = {
            "task_name": TaskType.summarization.value,
            "dataset_name": "cnndm",
            "metric_names": ["bleu"],
        }

        processor = get_processor(TaskType.summarization.value)

        sys_info = processor.process(metadata, data)

        # analysis.write_to_directory("./")
        self.assertIsNotNone(sys_info.results.fine_grained)
        self.assertGreater(len(sys_info.results.overall), 0)

    def test_default_features_dont_modify_condgen(self):

        condgen_processor = get_processor(TaskType.conditional_generation.value)
        sum_processor = get_processor(TaskType.summarization.value)

        condgen_features_1 = condgen_processor.default_features()
        sum_features = sum_processor.default_features()
        condgen_features_2 = condgen_processor.default_features()

        # MT features didn't change condgen features
        self.assertDictEqual(condgen_features_1, condgen_features_2)
        # condgen features are a subset of sum features
        self.assertDictEqual(sum_features, {**sum_features, **condgen_features_1})

    # Commented out following code since it's too slow for unittest
    @unittest.skipUnless('test_sum' in OPTIONAL_TEST_SUITES, reason='time consuming')
    def test_datalab_loader(self):

        json_output_customized = cache_api.cache_online_file(
            'http://www.phontron.com/download/cnndm-bart-output.txt',
            'predictions/summarization/cnndm-bart-output.txt',
        )

        loader = get_loader_class(TaskType.summarization).from_datalab(
            dataset=DatalabLoaderOption("cnn_dailymail", "3.0.0"),
            output_data=json_output_customized,
            output_source=Source.local_filesystem,
            output_file_type=FileType.text,
        )
        data = loader.load()

        metadata = {
            "task_name": TaskType.summarization.value,
            "dataset_name": "cnn_dailymail",
            "sub_dataset_name": "3.0.0",
            "metric_names": ["rouge1"],
        }
        processor = get_processor(TaskType.summarization)
        sys_info = processor.process(metadata, data)

        self.assertIsNotNone(sys_info.results.fine_grained)
        self.assertGreater(len(sys_info.results.overall), 0)

    def test_generate_system_human_eval(self):
        loader = get_loader_class(TaskType.summarization)(
            self.tsv_dataset,
            self.txt_output,
            Source.local_filesystem,
            Source.local_filesystem,
            FileType.tsv,
            FileType.text,
        )
        data = loader.load()

        metadata = {
            "task_name": TaskType.summarization.value,
            "dataset_name": "cnndm",
            "metric_configs": [
                ExternalEvalConfig(
                    name="LikertScore_fluency",
                    aspect="fluency",
                    n_annotators=2,
                    categories=5,
                    external_stats=np.array([[2, 2], [1, 1], [3, 3]]),
                )
            ],
        }

        processor = get_processor(TaskType.summarization.value)

        sys_info = processor.process(metadata, data)
        # print(sys_info.results.overall)
        # print(metadata["metric_configs"][0])
        self.assertIsNotNone(sys_info.results.fine_grained)
        self.assertEqual(sys_info.results.overall["LikertScore_fluency"].value, 2.0)
        self.assertEqual(sys_info.results.overall["LikertScore_fluency"].agreement, 1.0)
