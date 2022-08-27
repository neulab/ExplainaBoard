import os
import unittest

from integration_tests.utils import test_artifacts_path

from explainaboard import FileType, get_processor, Source, TaskType
from explainaboard.loaders.file_loader import DatalabLoaderOption
from explainaboard.loaders.loader_registry import get_loader_class


class GrammarErrorCorrectionTest(unittest.TestCase):
    artifact_path = os.path.join(test_artifacts_path, "gaokao")
    json_output = os.path.join(artifact_path, "rst_2018_quanguojuan1_gec.json")

    def test_datalab_loader(self):
        loader = get_loader_class(TaskType.grammatical_error_correction).from_datalab(
            dataset=DatalabLoaderOption("gaokao2018_np1", "writing-grammar"),
            output_data=self.json_output,
            output_source=Source.local_filesystem,
            output_file_type=FileType.json,
        )
        data = loader.load().samples

        metadata = {
            "task_name": TaskType.grammatical_error_correction.value,
            "dataset_name": "gaokao2018_np1",
            "sub_dataset_name": "writing-grammar",
            "metric_names": ["SeqCorrectCount"],
        }
        processor = get_processor(TaskType.grammatical_error_correction.value)
        sys_info = processor.process(metadata, data)
        overall_map = {x.metric_name: x for x in sys_info.results.overall[0]}
        self.assertAlmostEqual(overall_map["SeqCorrectCount"].value, 8)
        self.assertIsNotNone(sys_info.results.analyses)


if __name__ == '__main__':
    unittest.main()
