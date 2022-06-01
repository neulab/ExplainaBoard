import os
import unittest

from explainaboard import FileType, get_processor, Source, TaskType
from explainaboard.loaders.file_loader import DatalabLoaderOption
from explainaboard.loaders.loader_registry import get_datalab_loader
from explainaboard.tests.utils import test_artifacts_path


class TestGrammarErrorCorrection(unittest.TestCase):
    artifact_path = os.path.join(test_artifacts_path, "gaokao")
    json_output = os.path.join(artifact_path, "rst_2018_quanguojuan1_gec.json")

    def test_datalab_loader(self):
        loader = get_datalab_loader(
            TaskType.grammatical_error_correction,
            dataset=DatalabLoaderOption("gaokao2018_np1", "writing-grammar"),
            output_data=self.json_output,
            output_source=Source.local_filesystem,
            output_file_type=FileType.json,
        )
        data = loader.load()

        metadata = {
            "task_name": TaskType.grammatical_error_correction.value,
            "dataset_name": "gaokao2018_np1",
            "sub_dataset_name": "writing-grammar",
            "metric_names": ["SeqCorrectCount"],
        }
        processor = get_processor(TaskType.grammatical_error_correction.value)
        sys_info = processor.process(metadata, data)
        processor.print_bucket_info(sys_info.results.fine_grained)
        # print(sys_info.results.overall["SeqCorrectCount"].value)
        self.assertAlmostEqual(sys_info.results.overall["SeqCorrectCount"].value, 8)
        self.assertIsNotNone(sys_info.results.fine_grained)


if __name__ == '__main__':
    unittest.main()
