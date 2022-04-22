import os
import unittest

from explainaboard import FileType, get_processor, Source, TaskType
from explainaboard.loaders.file_loader import DatalabLoaderOption
from explainaboard.loaders.loader_registry import (
    get_custom_dataset_loader,
    get_datalab_loader,
)
from explainaboard.tests.utils import test_artifacts_path


class TestNER(unittest.TestCase):
    artifact_path = os.path.join(test_artifacts_path, "ner")
    conll_dataset = os.path.join(artifact_path, "dataset.tsv")
    conll_output = os.path.join(artifact_path, "output.tsv")
    conll_output_full = os.path.join(artifact_path, "conll2003-elmo-output.conll")

    def test_generate_system_analysis(self):
        loader = get_custom_dataset_loader(
            TaskType.named_entity_recognition,
            self.conll_dataset,
            self.conll_output,
            Source.local_filesystem,
            Source.local_filesystem,
            FileType.conll,
            FileType.conll,
        )
        data = loader.load()

        metadata = {
            "task_name": TaskType.named_entity_recognition.value,
            # "dataset_name": "conll2003",
            # "sub_dataset_name":"ner",
            "metric_names": ["F1Score"],
        }
        processor = get_processor(TaskType.named_entity_recognition)
        sys_info = processor.process(metadata, data)

        self.assertIsNotNone(sys_info.results.fine_grained)
        self.assertGreater(len(sys_info.results.overall), 0)

        # ------ Deep Test --------

        # test: training set dependent features should be disabled when
        # training dataset is not provided
        activate_features = sys_info.results.fine_grained.keys()
        self.assertTrue(
            "span_econ" not in activate_features
            and "span_efre" not in activate_features
        )

    def test_datalab_loader(self):
        loader = get_datalab_loader(
            TaskType.named_entity_recognition,
            dataset=DatalabLoaderOption("conll2003", "ner"),
            output_data=self.conll_output_full,
            output_source=Source.local_filesystem,
            output_file_type=FileType.conll,
        )
        data = loader.load()

        metadata = {
            "task_name": TaskType.named_entity_recognition.value,
            "dataset_name": "conll2003",
            "sub_dataset_name": "ner",
            "metric_names": ["F1Score"],
        }
        processor = get_processor(TaskType.named_entity_recognition)
        sys_info = processor.process(metadata, data)

        self.assertIsNotNone(sys_info.results.fine_grained)
        self.assertGreater(len(sys_info.results.overall), 0)

        # ---------------------------------------------------------------------------
        #                               Deep Test
        # ---------------------------------------------------------------------------

        # 1. Unittest: training set dependent features shouldn't included
        # when training dataset is not provided
        activate_features = sys_info.results.fine_grained.keys()
        self.assertTrue(
            "span_econ" in activate_features and "span_efre" in activate_features
        )

        # 2. Unittest: test the number of buckets of training dependent features
        n_buckets = len(sys_info.results.fine_grained["span_econ"])
        self.assertEqual(n_buckets, 3)

        # 3. Unittest: test detailed bucket information: bucket interval
        second_interval = list(sys_info.results.fine_grained["span_econ"].keys())[
            1
        ]  # [0.007462686567164179,0.9565217391304348]
        self.assertAlmostEqual(
            second_interval[0],
            0.007462686567164179,
            4,
            "almost equal",
        )
        self.assertAlmostEqual(
            second_interval[1],
            0.8603351955307262,
            4,
            "almost equal",
        )
        # 4. Unittest: test detailed bucket information: bucket samples
        self.assertEqual(
            sys_info.results.fine_grained["span_econ"][second_interval].n_samples, 999
        )

        # 5. Unittest: test detailed bucket information: metric
        self.assertEqual(
            sys_info.results.fine_grained["span_econ"][second_interval]
            .performances[0]
            .metric_name,
            "F1",
        )
        self.assertAlmostEqual(
            sys_info.results.fine_grained["span_econ"][second_interval]
            .performances[0]
            .value,
            0.9273461150353179,
            4,
            "almost equal",
        )
        # 6 Unittest: test detailed bucket information: confidence interval
        self.assertGreater(
            sys_info.results.fine_grained["span_econ"][second_interval]
            .performances[0]
            .confidence_score_low,
            0,
        )

        # 7. Unittest: test if only error cases are printed (this is the expected case,
        # especially for sequence labeling tasks. Otherwise, the analysis report files
        # will be too large.)
        self.assertLess(
            len(
                sys_info.results.fine_grained["span_econ"][
                    second_interval
                ].bucket_samples
            ),
            sys_info.results.fine_grained["span_econ"][second_interval].n_samples,
        )

        # 8. Unittest: customized metadata (TODO(Pengfei):
        #  lacks implementation of dataloader?)

        # 9. Unittest: customized features (TODO(Pengfei):
        #  lacks implementation of dataloader?)
