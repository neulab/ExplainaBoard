import dataclasses
import os
import unittest

from explainaboard import FileType, get_processor, Source, TaskType
from explainaboard.loaders.loader_registry import get_custom_dataset_loader
from explainaboard.tests.utils import test_artifacts_path


class TestMachineTranslation(unittest.TestCase):
    artifact_path = os.path.join(test_artifacts_path, "machine_translation")
    tsv_dataset = os.path.join(artifact_path, "dataset.tsv")
    txt_output = os.path.join(artifact_path, "output.txt")
    json_output_with_features = os.path.join(artifact_path, "output_with_features.json")

    def test_load_tsv(self):
        loader = get_custom_dataset_loader(
            TaskType.machine_translation,
            self.tsv_dataset,
            self.txt_output,
            Source.local_filesystem,
            Source.local_filesystem,
            FileType.tsv,
            FileType.text,
        )
        data = loader.load()
        self.assertEqual(len(data), 4)
        self.assertEqual(
            data[0],
            {
                'source': 'Ak sa chcete dostať ešte hlbšie, môžete si všimnúť '
                + 'trhlinky.',
                'reference': 'Now just to get really deep in , you can really get to '
                + 'the cracks .',
                'id': '0',
                'hypothesis': 'If you want to get a deeper , you can see the forces .',
            },
        )

    def test_generate_system_analysis(self):
        loader = get_custom_dataset_loader(
            TaskType.machine_translation,
            self.tsv_dataset,
            self.txt_output,
            Source.local_filesystem,
            Source.local_filesystem,
            FileType.tsv,
            FileType.text,
        )
        data = loader.load()

        metadata = {
            "task_name": TaskType.machine_translation.value,
            "dataset_name": "ted_multi",
            "metric_names": ["bleu"],
        }

        processor = get_processor(TaskType.machine_translation.value)

        sys_info = processor.process(metadata, data)

        # analysis.write_to_directory("./")
        self.assertIsNotNone(sys_info.results.fine_grained)
        self.assertGreater(len(sys_info.results.overall), 0)

    def test_default_features_dont_modify_condgen(self):

        condgen_processor = get_processor(TaskType.conditional_generation.value)
        mt_processor = get_processor(TaskType.machine_translation.value)

        condgen_features_1 = condgen_processor.default_features()
        mt_features = mt_processor.default_features()
        condgen_features_2 = condgen_processor.default_features()

        # MT features didn't change condgen features
        self.assertDictEqual(condgen_features_1, condgen_features_2)
        # condgen features are a subset of MT features
        self.assertDictEqual(mt_features, {**mt_features, **condgen_features_1})

    def test_custom_features(self):
        loader = get_custom_dataset_loader(
            TaskType.machine_translation,
            self.tsv_dataset,
            self.json_output_with_features,
            Source.local_filesystem,
            Source.local_filesystem,
            FileType.tsv,
            FileType.json,
        )
        data = loader.load()
        self.assertEqual(len(data), 4)
        self.assertEqual(
            data[0],
            {
                'source': 'Ak sa chcete dostať ešte hlbšie, môžete si všimnúť '
                + 'trhlinky.',
                'reference': 'Now just to get really deep in , you can really get to '
                + 'the cracks .',
                'id': '0',
                'hypothesis': 'If you want to get a deeper , you can see the forces .',
                'num_capital_letters': 1,
            },
        )

        processor = get_processor(TaskType.machine_translation.value)

        sys_info = processor.process(dataclasses.asdict(data.metadata), data.samples)
        self.assertTrue('num_capital_letters' in sys_info.results.fine_grained)


if __name__ == '__main__':
    unittest.main()
