import unittest

from explainaboard import FileType, get_processor, Source, TaskType
from explainaboard.loaders.file_loader import DatalabLoaderOption
from explainaboard.loaders.loader_registry import get_loader_class
from explainaboard.utils import cache_api


class TestQATableTextHybrid(unittest.TestCase):
    json_output_customized = cache_api.cache_online_file(
        'https://explainaboard.s3.amazonaws.com/system_outputs/'
        'qa_table_text_hybrid/predictions_list.json',
        'predictions/qa_table_text_hybrid/predictions_list.json',
    )

    def test_datalab_loader(self):
        loader = get_loader_class(TaskType.qa_table_text_hybrid).from_datalab(
            dataset=DatalabLoaderOption("tat_qa"),
            output_data=self.json_output_customized,
            output_source=Source.local_filesystem,
            output_file_type=FileType.json,
        )
        data = loader.load()

        metadata = {
            "task_name": TaskType.qa_table_text_hybrid,
            "dataset_name": "tat_qa",
            "metric_names": ["ExactMatchHybridQA", "F1ScoreHybridQA"],
        }
        processor = get_processor(TaskType.qa_table_text_hybrid)
        sys_info = processor.process(metadata, data)
        self.assertIsNotNone(sys_info.results.analyses)

        self.assertGreater(len(sys_info.results.overall), 0)
        self.assertAlmostEqual(
            sys_info.results.overall[0][0].value,
            0.746223,
            4,
            "almost equal",
        )


if __name__ == '__main__':
    unittest.main()
