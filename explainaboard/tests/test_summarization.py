import os
import pathlib
import unittest

from explainaboard import FileType, get_loader, get_processor, Source, TaskType

artifacts_path = os.path.dirname(pathlib.Path(__file__)) + "/artifacts/"


class TestSummarization(unittest.TestCase):
    def test_generate_system_analysis(self):
        """TODO: should add harder tests"""

        path_data = artifacts_path + "test-summ.tsv"
        loader = get_loader(
            TaskType.summarization, path_data, Source.local_filesystem, FileType.tsv
        )
        data = list(loader.load())

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
