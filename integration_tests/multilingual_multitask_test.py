from __future__ import annotations

import json
import os
import unittest

from integration_tests.utils import test_artifacts_path

from explainaboard import FileType, get_processor_class, TaskType
from explainaboard.loaders.loader_factory import get_loader_class
from explainaboard.utils.tensor_analysis import (
    aggregate_score_tensor,
    filter_score_tensor,
    print_score_tensor,
)


class MultilingualMultiTaskTest(unittest.TestCase):
    artifact_path = os.path.join(test_artifacts_path, "multilingual")

    @unittest.skip(
        reason="this unit test is broken but also complicated. it'd be better to fix "
        "it and also make it simpler"
    )
    def test_batch_processing(self):
        sys_out_dir = os.path.join(self.artifact_path, "CL-mt5base", "xnli")

        datasets = [
            os.path.join(sys_out_dir, "datasets", file)
            for file in os.listdir(os.path.join(sys_out_dir, "datasets"))
        ]

        outputs = [
            os.path.join(sys_out_dir, "outputs", file)
            for file in os.listdir(os.path.join(sys_out_dir, "outputs"))
        ]

        file_type = FileType.json
        task_dummy = TaskType.text_classification
        tasks = []
        for dataset, output in zip(datasets, outputs):
            loader = get_loader_class(task_dummy)(
                dataset,
                output,
                dataset_file_type=file_type,
                output_file_type=file_type,
            )
            if not loader.user_defined_metadata_configs:
                raise ValueError(
                    f"user_defined_metadata_configs in system output {output} hasn't "
                    "been specified or task name should be specified"
                )
            tasks.append(loader.user_defined_metadata_configs["task_name"])

        # Get loaders using real `task` and `file_type`
        loaders = [
            get_loader_class(task)(
                dataset,
                output,
                dataset_file_type=file_type,
                output_file_type=file_type,
            )
            for dataset, output, task in zip(datasets, outputs, tasks)
        ]
        system_outputs = [loader.load() for loader in loaders]

        # Run analysis
        reports = []
        metadata = {}
        for loader, system_output, task in zip(loaders, system_outputs, tasks):

            metadata.update(loader.user_defined_metadata_configs)

            report = get_processor_class(task)().process(
                metadata=metadata, sys_output=system_output
            )
            reports.append(report)

        self.assertEqual(len(reports), 2)

    def test_batch_meta_analysis(self):
        # Get reports
        path_reports_folder = os.path.join(test_artifacts_path, "reports")
        reports = [
            os.path.join(path_reports_folder, report_name)
            for report_name in os.listdir(path_reports_folder)
        ]

        score_tensor = {}
        for report in reports:
            with open(report) as fin:

                report_dict = json.load(fin)

                system_name = report_dict["system_name"]
                dataset_name = report_dict["dataset_name"]
                language = report_dict["language"]
                # TODO(Pengfei): So far, only one metric is considered
                metric = report_dict["metric_names"][0]
                score_info = report_dict["results"]["overall"][metric]

                if system_name not in score_tensor.keys():
                    score_tensor[system_name] = {}
                if dataset_name not in score_tensor[system_name].keys():
                    score_tensor[system_name][dataset_name] = {}
                if language not in score_tensor[system_name][dataset_name].keys():
                    score_tensor[system_name][dataset_name][language] = {}
                score_tensor[system_name][dataset_name][language] = score_info

        self.assertEqual(
            len(list(score_tensor.keys())), len(["CL-mlpp15out1sum", "CL-mt5base"])
        )
        print_score_tensor(score_tensor)

        # filter by three dimensions
        systems = ["CL-mlpp15out1sum"]
        datasets = ["marc", "xquad"]
        languages = ["en", "zh"]

        score_tensor_filter = filter_score_tensor(
            score_tensor, systems, datasets, languages
        )
        self.assertEqual(list(score_tensor_filter.keys()), ["CL-mlpp15out1sum"])
        print_score_tensor(score_tensor_filter)

        # aggregation by three dimensions
        systems_aggregation = None
        datasets_aggregation = None
        languages_aggregation = "average"
        score_tensor_aggregated = aggregate_score_tensor(
            score_tensor,
            systems_aggregation,
            datasets_aggregation,
            languages_aggregation,
        )
        print_score_tensor(score_tensor_aggregated)
        self.assertEqual(
            list(score_tensor_aggregated["CL-mt5base"]["xnli"].keys()),
            ["all_languages"],
        )

        # aggregation by three dimensions
        systems_aggregation = "minus"
        datasets_aggregation = None
        languages_aggregation = None
        score_tensor_aggregated = aggregate_score_tensor(
            score_tensor,
            systems_aggregation,
            datasets_aggregation,
            languages_aggregation,
        )
        print_score_tensor(score_tensor_aggregated)
        self.assertEqual(
            len(list(score_tensor_aggregated.keys())),
            len(["CL-mlpp15out1sum V.S CL-mt5base"]),
        )
