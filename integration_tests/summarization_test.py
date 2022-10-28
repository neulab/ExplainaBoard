from __future__ import annotations

import os
import unittest

from integration_tests.utils import OPTIONAL_TEST_SUITES, test_artifacts_path
import numpy as np

from explainaboard import FileType, get_processor_class, Source, TaskType
from explainaboard.loaders import get_loader_class
from explainaboard.loaders.file_loader import DatalabLoaderOption
from explainaboard.metrics.external_eval import ExternalEvalConfig
from explainaboard.metrics.metric import Score
from explainaboard.utils import cache_api


class SummarizationTest(unittest.TestCase):
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
        data = loader.load().samples

        metadata = {
            "task_name": TaskType.summarization.value,
            "dataset_name": "cnn_dailymail_custom",
            "metric_names": ["bleu"],
        }

        processor = get_processor_class(TaskType.summarization)()

        sys_info = processor.process(metadata, data, skip_failed_analyses=True)

        self.assertGreater(len(sys_info.results.analyses), 0)
        self.assertGreater(len(sys_info.results.overall), 0)

    def test_default_features_dont_modify_condgen(self):

        condgen_processor = get_processor_class(TaskType.conditional_generation)()
        sum_processor = get_processor_class(TaskType.summarization)()

        condgen_features_1 = condgen_processor.default_analysis_levels()
        sum_features = sum_processor.default_analysis_levels()
        condgen_features_2 = condgen_processor.default_analysis_levels()

        for cf1, cf2, sumf in zip(condgen_features_1, condgen_features_2, sum_features):
            lcf1 = set(cf1.features.keys())
            lcf2 = set(cf2.features.keys())
            lsumf = set(sumf.features.keys())
            self.assertEqual(lcf1, lcf2)
            # condgen features are a subset of MT features
            self.assertTrue(all([x in lsumf] for x in lcf1))

    # Commented out following code since it's too slow for unittest
    @unittest.skipUnless("test_sum" in OPTIONAL_TEST_SUITES, reason="time consuming")
    def test_datalab_loader(self):

        json_output_customized = cache_api.cache_online_file(
            "https://storage.googleapis.com/inspired-public-data/"
            "explainaboard/task_data/summarization/cnndm-bart-output.txt",
            "explainaboard/task_data/summarization/cnndm-bart-output.txt",
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
        processor = get_processor_class(TaskType.summarization)()
        sys_info = processor.process(metadata, data.samples)

        self.assertGreater(len(sys_info.results.analyses), 0)
        self.assertGreater(len(sys_info.results.overall), 0)

    @unittest.skip("Not yet fixed in v0.11")
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
            "metric_configs": {
                "example": [
                    ExternalEvalConfig(
                        name="LikertScore_fluency",
                        aspect="fluency",
                        n_annotators=2,
                        categories=5,
                        external_stats=np.array([[2, 2], [1, 1], [3, 3]]),
                    )
                ]
            },
        }

        processor = get_processor_class(TaskType.summarization)()

        sys_info = processor.process(metadata, data.samples)
        self.assertGreater(len(sys_info.results.analyses), 0)
        fluency = [
            metric_result
            for metric_name, metric_result in sys_info.results.overall[
                "example"
            ].items()
            if metric_name == "LikertScore_fluency"
        ][0]
        self.assertEqual(fluency.get_value(Score, "score").value, 2.0)
        self.assertEqual(fluency.get_value(Score, "agreement").value, 1.0)
