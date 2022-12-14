from __future__ import annotations

import dataclasses
import os
import unittest

from integration_tests.utils import test_artifacts_path

from explainaboard import FileType, get_processor_class, Source, TaskType
from explainaboard.loaders.loader_factory import get_loader_class


class MachineTranslationTest(unittest.TestCase):
    artifact_path = os.path.join(test_artifacts_path, "machine_translation")
    tsv_dataset = os.path.join(artifact_path, "dataset.tsv")
    txt_output = os.path.join(artifact_path, "output.txt")
    json_output_with_features = os.path.join(artifact_path, "output_with_features.json")

    def test_load_tsv(self):
        loader = get_loader_class(TaskType.machine_translation)(
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
                "source": "Ak sa chcete dostať ešte hlbšie, môžete si všimnúť "
                + "trhlinky.",
                "reference": "Now just to get really deep in , you can really get to "
                + "the cracks .",
                "id": "0",
                "hypothesis": "If you want to get a deeper , you can see the forces .",
            },
        )

    def test_generate_system_analysis(self):
        loader = get_loader_class(TaskType.machine_translation)(
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
            "metric_names": ["bleu"],
        }

        processor = get_processor_class(TaskType.machine_translation)()

        sys_info = processor.process(metadata, data, skip_failed_analyses=True)

        self.assertGreater(len(sys_info.results.analyses), 0)
        self.assertGreater(len(sys_info.results.overall), 0)

    def test_default_features_dont_modify_condgen(self):

        condgen_processor = get_processor_class(TaskType.conditional_generation)()
        mt_processor = get_processor_class(TaskType.machine_translation)()

        condgen_features_1 = condgen_processor.default_analysis_levels()
        mt_features = mt_processor.default_analysis_levels()
        condgen_features_2 = condgen_processor.default_analysis_levels()

        # MT features didn't change condgen features
        for cf1, cf2, mtf in zip(condgen_features_1, condgen_features_2, mt_features):
            lcf1 = set(cf1.features.keys())
            lcf2 = set(cf2.features.keys())
            lmtf = set(mtf.features.keys())
            self.assertEqual(lcf1, lcf2)
            # condgen features are a subset of MT features
            self.assertTrue(all([x in lmtf] for x in lcf1))

    def test_custom_features(self):
        loader = get_loader_class(TaskType.machine_translation)(
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
                "source": "Ak sa chcete dostať ešte hlbšie, môžete si všimnúť "
                + "trhlinky.",
                "reference": "Now just to get really deep in , you can really get to "
                + "the cracks .",
                "id": "0",
                "hypothesis": "If you want to get a deeper , you can see the forces .",
                "num_capital_letters": 1,
            },
        )

        processor = get_processor_class(TaskType.machine_translation)()

        sys_info = processor.process(
            {
                f.name: getattr(data.metadata, f.name)
                for f in dataclasses.fields(data.metadata)
            },
            data.samples,
            skip_failed_analyses=True,
        )
        analysis_names = [x.name for x in sys_info.results.analyses if x is not None]
        self.assertIn("num_capital_letters", analysis_names)


if __name__ == "__main__":
    unittest.main()
