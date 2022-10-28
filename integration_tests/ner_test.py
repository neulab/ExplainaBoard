from __future__ import annotations

import dataclasses
import os
import unittest

from integration_tests.utils import test_artifacts_path

from explainaboard import FileType, get_processor_class, Source, TaskType
from explainaboard.analysis.analyses import AnalysisResult, BucketAnalysisDetails
from explainaboard.loaders.file_loader import DatalabLoaderOption
from explainaboard.loaders.loader_factory import get_loader_class
from explainaboard.metrics.metric import ConfidenceInterval, Score
from explainaboard.utils import cache_api
from explainaboard.utils.typing_utils import narrow, unwrap


class NERTest(unittest.TestCase):
    artifact_path = os.path.join(test_artifacts_path, "ner")
    conll_dataset = os.path.join(artifact_path, "dataset.tsv")
    conll_output = os.path.join(artifact_path, "output.tsv")
    conll_output_full = os.path.join(artifact_path, "conll2003-elmo-output.conll")

    json_output_customized = cache_api.cache_online_file(
        "https://storage.googleapis.com/inspired-public-data/"
        "explainaboard/task_data/named_entity_recognition/test-conll03.json",
        "explainaboard/task_data/named_entity_recognition/test-conll03.json",
    )

    def test_generate_system_analysis(self):
        loader = get_loader_class(TaskType.named_entity_recognition)(
            self.conll_dataset,
            self.conll_output,
            Source.local_filesystem,
            Source.local_filesystem,
            FileType.conll,
            FileType.conll,
        )
        data = loader.load().samples

        metadata = {
            "task_name": TaskType.named_entity_recognition.value,
            # "dataset_name": "conll2003",
            # "sub_dataset_name":"ner",
            "metric_names": ["F1Score"],
        }
        processor = get_processor_class(TaskType.named_entity_recognition)()
        sys_info = processor.process(metadata, data, skip_failed_analyses=True)

        self.assertGreater(len(sys_info.results.analyses), 0)
        self.assertGreater(len(sys_info.results.overall), 0)

        # ------ Deep Test --------

        # test: training set dependent features should be disabled when
        # training dataset is not provided
        activate_features = [x.name for x in sys_info.results.analyses if x is not None]
        self.assertTrue("span_econ" not in activate_features)
        self.assertTrue("span_efre" not in activate_features)

    def test_datalab_loader(self):
        loader = get_loader_class(TaskType.named_entity_recognition).from_datalab(
            dataset=DatalabLoaderOption("conll2003", "ner"),
            output_data=self.conll_output_full,
            output_source=Source.local_filesystem,
            output_file_type=FileType.conll,
        )
        data = loader.load().samples

        metadata = {
            "task_name": TaskType.named_entity_recognition.value,
            "dataset_name": "conll2003",
            "sub_dataset_name": "ner",
            "metric_names": ["F1Score"],
        }
        processor = get_processor_class(TaskType.named_entity_recognition)()
        sys_info = processor.process(metadata, data)

        self.assertGreater(len(sys_info.results.analyses), 0)
        self.assertGreater(len(sys_info.results.overall), 0)

        # ---------------------------------------------------------------------------
        #                               Deep Test
        # ---------------------------------------------------------------------------

        # 1. Unittest: training set dependent features shouldn't be included
        # when training dataset is not provided
        span_analysis_map = {
            x.name: x for x in sys_info.results.analyses if x is not None
        }
        self.assertTrue("span_econ" in span_analysis_map)
        self.assertTrue("span_efre" in span_analysis_map)

        # 2. Unittest: test the number of buckets of training dependent features
        analysis = narrow(AnalysisResult, span_analysis_map["span_econ"])
        details = narrow(BucketAnalysisDetails, analysis.details)
        self.assertEqual(len(details.bucket_performances), 3)

        # 3. Unittest: test detailed bucket information: bucket interval
        # [0.007462686567164179,0.9565217391304348]
        second_bucket = details.bucket_performances[1]
        second_bucket_interval = unwrap(second_bucket.bucket_interval)
        self.assertAlmostEqual(second_bucket_interval[0], 0.007462686567164179, 4)
        self.assertAlmostEqual(second_bucket_interval[1], 0.8571428571428571, 4)
        # 4. Unittest: test detailed bucket information: bucket samples
        self.assertEqual(second_bucket.n_samples, 1050)

        # 5. Unittest: test detailed bucket information: metric
        self.assertAlmostEqual(
            unwrap(second_bucket.results["F1"].get_value(Score, "score")).value,
            0.9121588089330025,
            4,
        )
        # 6 Unittest: test detailed bucket information: confidence interval
        for bucket_vals in sys_info.results.analyses:
            if not isinstance(bucket_vals.details, BucketAnalysisDetails):
                continue
            for bucket in bucket_vals.details.bucket_performances:
                for result in bucket.results.values():
                    ci = result.get_value_or_none(ConfidenceInterval, "score_ci")
                    if ci is not None:
                        value = unwrap(result.get_value(Score, "score")).value
                        self.assertGreaterEqual(value, ci.low)
                        self.assertGreaterEqual(ci.high, value)

        # 7. Unittest: test if only fewer cases are printed (this is the expected
        # case, especially for sequence labeling tasks. Otherwise, the analysis report
        # files will be too large.)
        self.assertLess(
            len(second_bucket.bucket_samples),
            second_bucket.n_samples,
        )

        # 8. Unittest: customized metadata (TODO(Pengfei):
        #  lacks implementation of dataloader?)

        # 9. Unittest: customized features (TODO(Pengfei):
        #  lacks implementation of dataloader?)

    def test_customized_metadata1(self):
        loader = get_loader_class(TaskType.named_entity_recognition).from_datalab(
            dataset=DatalabLoaderOption("conll2003", "ner"),
            output_data=self.json_output_customized,
            output_source=Source.local_filesystem,
            output_file_type=FileType.json,
        )
        data = loader.load()
        metadata = dataclasses.asdict(data.metadata)
        metadata.update(
            {
                "task_name": TaskType.named_entity_recognition.value,
            }
        )
        processor = get_processor_class(TaskType.named_entity_recognition)()
        sys_info = processor.process(metadata, data.samples)
        self.assertGreater(len(sys_info.results.analyses), 0)
        self.assertGreater(len(sys_info.results.overall), 0)
