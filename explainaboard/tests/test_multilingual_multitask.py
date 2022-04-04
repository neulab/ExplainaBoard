import json
import os
import pathlib
import unittest

from explainaboard import FileType, get_loader, get_processor, TaskType
from explainaboard.utils.tensor_analysis import (
    aggregate_score_tensor,
    filter_score_tensor,
    print_score_tensor,
)

artifacts_path = os.path.dirname(pathlib.Path(__file__)) + "/artifacts/"


class TestMultilingualMultiTask(unittest.TestCase):
    def test_batch_processing(self):

        path_system_output_folder = artifacts_path + "multilingual/CL-mt5base/xnli/"
        system_outputs = [
            path_system_output_folder + system_output_name
            for system_output_name in os.listdir(path_system_output_folder)
        ]

        file_type = FileType.json
        task_dummy = TaskType.text_classification
        task_system_outputs = []
        for x in system_outputs:
            loader = get_loader(task_dummy, data=x, file_type=file_type)
            if (
                loader.user_defined_metadata_configs is None
                or len(loader.user_defined_metadata_configs) == 0
            ):
                raise ValueError(
                    f"user_defined_metadata_configs in system output {x} has n't "
                    f"been specified or task name should be specified"
                )
            task_system_outputs.append(
                loader.user_defined_metadata_configs['task_name']
            )

        # Get loaders using real `task` and `file_type`
        loaders = [
            get_loader(task, data=x, file_type=FileType.json)
            for x, task in zip(system_outputs, task_system_outputs)
        ]
        system_datasets = [list(loader.load()) for loader in loaders]

        # Run analysis
        reports = []
        metadata = {}
        for loader, system_dataset, system_full_path, task in zip(
            loaders, system_datasets, system_outputs, task_system_outputs
        ):

            metadata.update(loader.user_defined_metadata_configs)

            report = get_processor(task).process(
                metadata=metadata, sys_output=system_dataset
            )
            reports.append(report)

        self.assertEqual(len(reports), 2)

    def test_batch_meta_analysis(self):

        # Get reports
        path_reports_folder = artifacts_path + "reports/"
        reports = [
            path_reports_folder + report_name
            for report_name in os.listdir(path_reports_folder)
        ]

        score_tensor = {}
        for report in reports:
            with open(report) as fin:

                report_dict = json.load(fin)

                model_name = report_dict["model_name"]
                dataset_name = report_dict["dataset_name"]
                language = report_dict["language"]
                # TODO(Pengfei): So far, only one metric is considered
                metric = report_dict["metric_names"][0]
                score_info = report_dict["results"]["overall"][metric]

                # print(model_name, dataset_name, language)

                if model_name not in score_tensor.keys():
                    score_tensor[model_name] = {}
                if dataset_name not in score_tensor[model_name].keys():
                    score_tensor[model_name][dataset_name] = {}
                if language not in score_tensor[model_name][dataset_name].keys():
                    score_tensor[model_name][dataset_name][language] = {}
                score_tensor[model_name][dataset_name][language] = score_info

        self.assertEqual(
            len(list(score_tensor.keys())), len(['CL-mlpp15out1sum', 'CL-mt5base'])
        )
        print_score_tensor(score_tensor)

        # filter by three dimensions
        models = ["CL-mlpp15out1sum"]
        datasets = ["marc", "xquad"]
        languages = ["en", "zh"]

        score_tensor_filter = filter_score_tensor(
            score_tensor, models, datasets, languages
        )
        self.assertEqual(list(score_tensor_filter.keys()), ['CL-mlpp15out1sum'])
        print_score_tensor(score_tensor_filter)

        # aggregation by three dimensions
        models_aggregation = None
        datasets_aggregation = None
        languages_aggregation = "average"
        score_tensor_aggregated = aggregate_score_tensor(
            score_tensor,
            models_aggregation,
            datasets_aggregation,
            languages_aggregation,
        )
        print(score_tensor_aggregated.keys())
        print_score_tensor(score_tensor_aggregated)
        self.assertEqual(
            list(score_tensor_aggregated["CL-mt5base"]["xnli"].keys()),
            ['all_languages'],
        )

        # aggregation by three dimensions
        models_aggregation = "minus"
        datasets_aggregation = None
        languages_aggregation = None
        score_tensor_aggregated = aggregate_score_tensor(
            score_tensor,
            models_aggregation,
            datasets_aggregation,
            languages_aggregation,
        )
        print(score_tensor_aggregated.keys())
        print_score_tensor(score_tensor_aggregated)
        self.assertEqual(
            list(score_tensor_aggregated.keys()), ['CL-mlpp15out1sum V.S CL-mt5base']
        )


if __name__ == '__main__':
    unittest.main()
