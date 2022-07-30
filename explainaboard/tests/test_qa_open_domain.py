import os
import unittest

from explainaboard import FileType, get_processor, Source, TaskType
from explainaboard.loaders.file_loader import DatalabLoaderOption
from explainaboard.loaders.loader_registry import get_loader_class
from explainaboard.tests.utils import test_artifacts_path


class TestQAOpenDomain(unittest.TestCase):
    artifact_path = os.path.join(test_artifacts_path, "qa_open_domain")
    system_output_file = os.path.join(artifact_path, "test.dpr.nq.txt")

    def test_datalab_loader(self):
        loader = get_loader_class(TaskType.qa_open_domain).from_datalab(
            dataset=DatalabLoaderOption("natural_questions_comp_gen"),
            output_data=self.system_output_file,
            output_source=Source.local_filesystem,
            output_file_type=FileType.text,
        )
        data = loader.load()

        metadata = {
            "task_name": TaskType.qa_open_domain.value,
            "dataset_name": "natural_questions_comp_gen",
            "metric_names": ["ExactMatchQA"],
        }
        processor = get_processor(TaskType.qa_open_domain.value)
        sys_info = processor.process(metadata, data)
        self.assertIsNotNone(sys_info.results.fine_grained)
