from __future__ import annotations

import unittest

from explainaboard import FileType, get_processor_class, Source, TaskType
from explainaboard.loaders.file_loader import DatalabLoaderOption
from explainaboard.loaders.loader_factory import get_loader_class
from explainaboard.metrics.metric import Score
from explainaboard.utils import cache_api


class QATableTextHybridTest(unittest.TestCase):
    json_output_customized = cache_api.cache_online_file(
        "https://explainaboard.s3.amazonaws.com/system_outputs/"
        "qa_table_text_hybrid/predictions_list.json",
        "predictions/qa_table_text_hybrid/predictions_list.json",
    )

    def test_datalab_loader(self):
        loader = get_loader_class(TaskType.qa_tat).from_datalab(
            dataset=DatalabLoaderOption("tat_qa"),
            output_data=self.json_output_customized,
            output_source=Source.local_filesystem,
            output_file_type=FileType.json,
        )
        data = loader.load()

        metadata = {
            "task_name": TaskType.qa_tat,
            "dataset_name": "tat_qa",
            "metric_names": ["ExactMatchQATat", "F1ScoreQATat"],
        }
        processor = get_processor_class(TaskType.qa_tat)()
        sys_info = processor.process(metadata, data)
        self.assertGreater(len(sys_info.results.analyses), 0)

        self.assertGreater(len(sys_info.results.overall), 0)
        self.assertAlmostEqual(
            sys_info.results.overall["example"]["ExactMatchQATat"]
            .get_value(Score, "score")
            .value,
            0.746978,
            3,
        )


if __name__ == "__main__":
    unittest.main()
