import os
import unittest

from integration_tests.utils import test_artifacts_path

from explainaboard import FileType, get_loader_class, get_processor, Source, TaskType


class NLGMetaEvalTest(unittest.TestCase):
    artifact_path = os.path.join(test_artifacts_path, "nlg_meta_evaluation")
    tsv_dataset = os.path.join(artifact_path, "./wmt20-DA/cs-en/data.tsv")
    txt_output = os.path.join(artifact_path, "./wmt20-DA/cs-en/score.txt")

    def test_da_cs_en(self):

        metadata = {
            "task_name": TaskType.nlg_meta_evaluation.value,
            "metric_names": ["SysPearsonCorr"],
            "confidence_alpha": None,
        }
        loader = get_loader_class(TaskType.nlg_meta_evaluation)(
            self.tsv_dataset,
            self.txt_output,
            Source.local_filesystem,
            Source.local_filesystem,
            FileType.tsv,
            FileType.text,
        )
        data = loader.load().samples
        processor = get_processor(TaskType.nlg_meta_evaluation)

        sys_info = processor.process(metadata, data)

        self.assertIsNotNone(sys_info.results.analyses)
        self.assertGreater(len(sys_info.results.overall), 0)
