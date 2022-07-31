import os
import unittest

from explainaboard import FileType, get_processor, Source, TaskType
from explainaboard.loaders.file_loader import DatalabLoaderOption
from explainaboard.loaders.loader_registry import get_loader_class
from explainaboard.tests.utils import test_artifacts_path


class TestCustomized(unittest.TestCase):
    artifact_path = os.path.join(test_artifacts_path, "text_classification")
    tsv_dataset = os.path.join(artifact_path, "dataset.tsv")
    txt_output = os.path.join(artifact_path, "output.txt")
    json_dataset = os.path.join(artifact_path, "dataset.json")
    json_output = os.path.join(artifact_path, "output_user_metadata.json")

    def test_custom_feature_function(self):
        loader = get_loader_class(TaskType.text_classification).from_datalab(
            dataset=DatalabLoaderOption(
                "sst2",
                custom_features={
                    "long_text_50": {
                        "dtype": "string",
                        "description": "whether a text is long",
                        "num_buckets": 2,
                        "func": "lambda x:'Long Text' if "
                        "len(x['text'].split()) > 50 "
                        "else 'Short Text'",
                    }
                },
            ),
            output_data=os.path.join(self.artifact_path, "output_sst2.txt"),
            output_source=Source.local_filesystem,
            output_file_type=FileType.text,
        )
        data = loader.load()
        metadata = {
            "task_name": TaskType.text_classification.value,
            "dataset_name": "sst2",
            "metric_names": ["Accuracy"],
            # don't forget this, otherwise the user-defined features will be ignored
            "custom_features": data.metadata.custom_features,
        }

        processor = get_processor(TaskType.text_classification.value)

        sys_info = processor.process(metadata, data.samples)
        processor.print_bucket_info(sys_info.results.fine_grained)

        self.assertEqual(len(data), 1821)
