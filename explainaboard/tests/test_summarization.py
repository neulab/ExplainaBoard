import os
import unittest

from explainaboard import FileType, get_processor, Source, TaskType
from explainaboard.loaders import get_custom_dataset_loader
from explainaboard.tests.utils import test_artifacts_path


class TestSummarization(unittest.TestCase):
    artifact_path = os.path.join(test_artifacts_path, "summarization")
    tsv_dataset = os.path.join(artifact_path, "dataset.tsv")
    txt_output = os.path.join(artifact_path, "output.txt")

    def test_load_tsv(self):
        loader = get_custom_dataset_loader(
            TaskType.summarization,
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
        loader = get_custom_dataset_loader(
            TaskType.summarization,
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


if __name__ == '__main__':
    unittest.main()
