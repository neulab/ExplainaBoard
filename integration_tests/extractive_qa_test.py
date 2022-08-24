import os
import unittest

from integration_tests.utils import test_artifacts_path

from explainaboard import FileType, get_processor, Source, TaskType
from explainaboard.loaders.loader_registry import get_loader_class
from explainaboard.utils.logging import get_logger


class ExtractiveQATest(unittest.TestCase):
    artifact_path = os.path.join(test_artifacts_path, "extractive_qa")

    def test_extractive_qa_en(self):
        json_en_dataset = os.path.join(self.artifact_path, "dataset-xquad-en.json")
        json_en_output = os.path.join(self.artifact_path, "output-xquad-en.json")
        loader = get_loader_class(TaskType.qa_extractive)(
            json_en_dataset,
            json_en_output,
            Source.local_filesystem,
            Source.local_filesystem,
            FileType.json,
            FileType.json,
        )
        data = loader.load().samples
        self.assertEqual(len(data), 1190)
        sample = data[0]
        self.assertEqual(sample["predicted_answers"], {"text": "308"})
        self.assertEqual(sample["id"], "0")
        self.assertEqual(sample["answers"], {"answer_start": [-1], "text": ["308"]})
        self.assertEqual(
            sample["question"], "How many points did the Panthers defense surrender ?"
        )
        self.assertTrue(sample["context"].startswith("The Panthers"))

        metadata = {
            "task_name": TaskType.qa_extractive,
            "dataset_name": "squad",
            "metric_names": ["F1ScoreQA", "ExactMatchQA"],
            # "language":"en"
        }

        processor = get_processor(TaskType.qa_extractive)
        sys_info = processor.process(metadata, data)

        self.assertIsNotNone(sys_info.results.analyses)
        self.assertGreater(len(sys_info.results.overall), 0)
        get_logger('test').info(f'OVERALL={sys_info.results.overall}')
        # should be 0.6974789915966386
        overall_map = {x.metric_name: x for x in sys_info.results.overall[0]}
        self.assertAlmostEqual(
            overall_map["ExactMatch"].value,
            0.6974789915966386,
            2,
            "almost equal",
        )
        # should be 0.8235975260931867
        self.assertAlmostEqual(
            overall_map["F1"].value,
            0.8235975260931867,
            2,
            "almost equal",
        )

    def test_extractive_qa_zh(self):
        json_zh_dataset = os.path.join(self.artifact_path, "dataset-xquad-zh.json")
        json_zh_output = os.path.join(self.artifact_path, "output-xquad-zh.json")
        loader = get_loader_class(TaskType.qa_extractive)(
            json_zh_dataset,
            json_zh_output,
            Source.local_filesystem,
            Source.local_filesystem,
            FileType.json,
            FileType.json,
        )
        data = loader.load().samples
        metadata = {
            "task_name": TaskType.qa_extractive.value,
            "dataset_name": "squad",
            "metric_names": ["F1Score", "ExactMatch"],
            "source_language": "zh",
            "target_language": "zh",
        }

        processor = get_processor(TaskType.qa_extractive)

        sys_info = processor.process(metadata, data)

        self.assertIsNotNone(sys_info.results.analyses)
        self.assertGreater(len(sys_info.results.overall), 0)
        # 0.6285714285714286
        overall_map = {x.metric_name: x for x in sys_info.results.overall[0]}
        self.assertAlmostEqual(
            overall_map["ExactMatch"].value,
            0.6285714285714286,
            2,
            "almost equal",
        )
        # 0.7559651817716333
        self.assertAlmostEqual(
            overall_map["F1"].value,
            0.7559651817716333,
            2,
            "almost equal",
        )
