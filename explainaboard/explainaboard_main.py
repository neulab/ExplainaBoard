import argparse
import json
import os
from tqdm import tqdm
from typing import Optional
import numpy as np
from explainaboard.analyzers.draw_hist import draw_bar_chart_from_report
import copy

from explainaboard import (
    get_loader,
    get_pairwise_performance_gap,
    get_processor,
    TaskType,
    FileType,
)



def aggregate_score_tensor(score_tensor:dict, models_aggregation:Optional[str] = None, datasets_aggregation:Optional[str] = None, languages_aggregation:Optional[str]= None):
    """
    This function aggregate score tensor based on specified parameters along three dimensions: model, dataset and language
    TODO(Pengfei):
     (1) this function could be duplicated
     (2) the way to implement the aggregation should be generalized

    :param score_tensor:
    :param models_aggregation:
    :param datasets_aggregation:
    :param languages_aggregation:
    :return:
    """


    if datasets_aggregation is not None and languages_aggregation is None:
        languages_aggregation = datasets_aggregation

    # Regarding language aggregation
    score_tensor_aggregated_language = {}

    if languages_aggregation == "average":
        for model_name, m_value in score_tensor.items():
            score_tensor_aggregated_language[model_name] = {}
            for dataset_name, d_value in score_tensor[model_name].items():
                score_info_template = copy.deepcopy(list(score_tensor[model_name][dataset_name].values())[0])

                # print(score_info_template)
                score_tensor_aggregated_language[model_name][dataset_name] = {}
                aggregated_score = np.average([score["value"] for score in score_tensor[model_name][dataset_name].values()])
                score_info_template["value"] = aggregated_score
                score_tensor_aggregated_language[model_name][dataset_name]["all_languages"] = score_info_template


    # Regarding dataset aggregation
    score_tensor_aggregated_dataset = {}
    if datasets_aggregation == "average":
        for model_name, m_value in score_tensor.items():
            score_tensor_aggregated_dataset[model_name] = {}
            aggregated_score = 0.0
            score_info_template = {}
            for dataset_name, d_value in score_tensor[model_name].items():
                score_info_template = copy.deepcopy(list(score_tensor[model_name][dataset_name].values())[0])
                aggregated_score += score_tensor_aggregated_language[model_name][dataset_name]["all_languages"]["value"]
            aggregated_score /= len(score_tensor[model_name].items())
            score_info_template["value"] = aggregated_score
            score_tensor_aggregated_dataset[model_name]["all_datasets"] = {}
            score_tensor_aggregated_dataset[model_name]["all_datasets"]["all_languages"] = score_info_template



    # Regarding model aggregation
    if datasets_aggregation is not None:
        score_tensor = score_tensor_aggregated_dataset
    elif languages_aggregation  is not None:
        score_tensor = score_tensor_aggregated_language



    score_tensor_aggregated_model = {}
    if models_aggregation == "minus":
        if len(score_tensor)!=2:
            raise ValueError("the number of systems should two")
        sys1_name, sys2_name = list(score_tensor.keys())
        score_tensor_aggregated_model[f"{sys1_name} V.S {sys2_name}"] = {}
        sys1_datasets = score_tensor[sys1_name].keys()
        sys2_datasets = score_tensor[sys2_name].keys()
        common_datasets = list(set(sys1_datasets) & set(sys2_datasets))
        for common_dataset in common_datasets:
            sys1_languages = score_tensor[sys1_name][common_dataset]
            sys2_languages = score_tensor[sys2_name][common_dataset]
            common_languages = list(set(sys1_languages) & set(sys2_languages))
            score_tensor_aggregated_model[f"{sys1_name} V.S {sys2_name}"][common_dataset] = {}

            for common_language in common_languages:
                score_tensor_aggregated_model[f"{sys1_name} V.S {sys2_name}"][common_dataset][common_language] = score_tensor[sys1_name][common_dataset][common_language]
                aggregated_score = score_tensor[sys1_name][common_dataset][common_language]["value"] - score_tensor[sys2_name][common_dataset][common_language]["value"]
                score_tensor_aggregated_model[f"{sys1_name} V.S {sys2_name}"][common_dataset][common_language]["value"] = aggregated_score




    if models_aggregation is not None:
        return score_tensor_aggregated_model
    elif datasets_aggregation is not None:
        return score_tensor_aggregated_dataset
    elif languages_aggregation is not None:
        return score_tensor_aggregated_language
    else:
        return score_tensor





def filter_score_tensor(score_tensor: dict, models:Optional[list], datasets:Optional[list], languages:Optional[list]):
    """
    filter score tensor based on given models, datasets and languages
    """
    score_tensor_copy = copy.deepcopy(score_tensor)

    score_tensor_filter = {}
    for model_name, m_value in score_tensor.items():
        if models is not None and model_name not in models:
            continue
        score_tensor_filter[model_name] = {}
        for dataset_name, d_value in score_tensor[model_name].items():
            if datasets is not None and dataset_name not in datasets:
                continue
            score_tensor_filter[model_name][dataset_name] = {}
            for language_name, l_value in score_tensor[model_name][dataset_name].items():
                if languages is not None and language_name not in languages:
                    continue
                score_tensor_filter[model_name][dataset_name][language_name] = score_tensor_copy[model_name][dataset_name][
                    language_name]
    return score_tensor_filter


def print_score_tensor(score_tensor:dict):
    """
    print the score_tensor, for example,
     ----------------------------------------
    Model: CL-mt5base, Dataset: xnli
    Language:       ar      bg      de      el      en      es      fr      hi      ru      sw      th      tr      ur      vi      zh
    Accuracy:       0.679   0.714   0.721   0.722   0.768   0.738   0.721   0.658   0.713   0.630   0.690   0.683   0.621   0.658   0.712

    ----------------------------------------
    Model: CL-mlpp15out1sum, Dataset: xnli
    Language:       ar      bg      de      el      en      es      fr      hi      ru      sw      th      tr      ur      vi      zh
    Accuracy:       0.696   0.739   0.735   0.739   0.787   0.768   0.730   0.682   0.725   0.660   0.710   0.705   0.657   0.692   0.731

    ----------------------------------------
    Model: CL-mlpp15out1sum, Dataset: marc
    Language:       de      en      es      fr      ja      zh
    Accuracy:       0.933   0.915   0.934   0.926   0.915   0.871

    """
    for model_name, m_value in score_tensor.items():
        for dataset_name, d_value in score_tensor[model_name].items():
            info_printed = f"----------------------------------------\nModel: {model_name}, Dataset: {dataset_name} \n"
            info_printed += f"Language:\t" + "\t".join(score_tensor[model_name][dataset_name].keys()) + "\n"
            metric_name = list(score_tensor[model_name][dataset_name].values())[0]["metric_name"]
            info_printed += f"{metric_name}:\t" + "\t".join(['{:.3f}'.format(score["value"]) for score in score_tensor[model_name][dataset_name].values()]) + "\n"
            print(info_printed)



def main():

    parser = argparse.ArgumentParser(description='Explainable Leaderboards for NLP')

    parser.add_argument('--task', type=str, required=False, help="the task name")

    parser.add_argument(
        '--system_outputs',
        type=str,
        required=True,
        nargs="+",
        help="the directories of system outputs. Multiple one should be separated by space, for example: system1 system2",
    )


	parser.add_argument(
        '--reports',
        type=str,
        required=False,
        nargs="+",
        help="the directories of analysis reports. Multiple one should be separated by space, for example: report1 report2",
    )


    parser.add_argument(
        '--models',
        type=str,
        required=False,
        nargs="+",
        help="the list of model names",
    )

    parser.add_argument(
        '--datasets',
        type=str,
        required=False,
        nargs="+",
        help="the list of dataset names",
    )

    parser.add_argument(
        '--languages',
        type=str,
        required=False,
        nargs="+",
        help="the list of language names",
    )

    parser.add_argument(
        '--models_aggregation',
        type=str,
        required=False,
        help="None|minus|combination",
    )

    parser.add_argument(
        '--datasets_aggregation',
        type=str,
        required=False,
        help="None|average|",
    )

    parser.add_argument(
        '--languages_aggregation',
        type=str,
        required=False,
        help="None|average|",
    )


    parser.add_argument(
        '--type',
        type=str,
        required=False,
        default="single",
        help="analysis type: single|pair|combine",
    )

    parser.add_argument(
        '--dataset',
        type=str,
        required=False,
        default=None,
        help="the name of dataset",
    )

    parser.add_argument(
        '--sub_dataset',
        type=str,
        required=False,
        default=None,
        help="the name of sub-dataset",
    )


    parser.add_argument(
        '--language',
        type=str,
        required=False,
        default="en",
        help="the language of system output",
    )

    parser.add_argument(
        '--reload_stat',
        type=str,
        required=False,
        default=None,
        help="reload precomputed statistics over training set (if exists)",
    )

    parser.add_argument(
        '--metrics',
        type=str,
        required=False,
        nargs="*",
        help="multiple metrics should be separated by space",
    )

    parser.add_argument(
        '--file_type',
        type=str,
        required=False,
        default=None,
        help="the file type: json, tsv, conll",
    )


	parser.add_argument(
        '--conf_value',
        type=float,
        required=False,
        default=0.05,
        help="the p-value with which to calculate the confidence interval",
    )


    parser.add_argument(
        '--output_dir',
        type=str,
        required=False,
        default="output",
        help="the directory of output files",
    )



    args = parser.parse_args()

    dataset = args.dataset
    sub_dataset = args.sub_dataset
    language = args.language
    task = args.task
    reload_stat = False if args.reload_stat == "0" else True
    system_outputs = args.system_outputs
    #num_outputs = len(system_outputs)
    metric_names = args.metrics
    file_type = args.file_type

    reports = args.reports
    output_dir = args.output_dir
    models = args.models
    datasets = args.datasets
    languages = args.languages
    models_aggregation = args.models_aggregation
    datasets_aggregation = args.datasets_aggregation
    languages_aggregation = args.languages_aggregation




    # If reports have been specified, ExplainaBoard cli will performan analysis over report files.
    if reports is not None:

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
                if language not in  score_tensor[model_name][dataset_name].keys():
                    score_tensor[model_name][dataset_name][language] = {}
                score_tensor[model_name][dataset_name][language] = score_info


        # print(json.dumps(score_tensor))


        score_tensor_filter = filter_score_tensor(score_tensor, models, datasets, languages)
        # print_score_tensor(score_tensor_filter)


        score_tensor_aggregated = aggregate_score_tensor(score_tensor_filter, models_aggregation, datasets_aggregation, languages_aggregation)
        print_score_tensor(score_tensor_aggregated)

        return True


    # Setup for generated reports and figures
    output_dir_figures = output_dir + "/" + "figures"
    output_dir_reports = output_dir + "/" + "reports"



    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_dir_figures):
        os.makedirs(output_dir_figures)
    if not os.path.exists(output_dir_reports):
        os.makedirs(output_dir_reports)




    # check for benchmark submission: explainaboard  --system_outputs ./data/system_outputs/sst2/user_specified_metadata.json
    if task is None:
        file_type = "json"
        task = TaskType.text_classification
        # TaskType.text_classification is set for temporal use, and this need to be generalized
        for x in system_outputs:
            loader = get_loader(task, data=x, file_type=file_type)
            if loader.user_defined_metadata_configs is None or len(loader.user_defined_metadata_configs) == 0:
                raise ValueError(f"user_defined_metadata_configs in system output {x} has n't been specified or task name should be specified")
            task = loader.user_defined_metadata_configs['task_name']



    # Checks on other inputs
    # if num_outputs > 2:
    #     raise ValueError(
    #         f'ExplainaBoard currently only supports 1 or 2 system outputs, but received {num_outputs}'
    #     )
    if task not in TaskType.list():
        raise ValueError(
            f'Task name {task} was not recognized. ExplainaBoard currently supports: {TaskType.list()}'
        )

    if file_type not in FileType.list():
        raise ValueError(
            f'File type {file_type} was not recognized. ExplainaBoard currently supports: {FileType.list()}'
        )



    # Read in data and check validity
    loads = []
    if file_type is not None:
        loaders = [get_loader(task, data=x, file_type = file_type) for x in system_outputs]
    else:
        loaders = [get_loader(task, data=x) for x in system_outputs] # use the default loaders that has been pre-defiend for each task
    system_datasets = [list(loader.load()) for loader in loaders]

    # validation
    if len(system_datasets) == 2:
        if len(system_datasets[0]) != len(system_datasets[1]):
            num0 = len(system_datasets[0])
            num1 = len(system_datasets[1])
            raise ValueError(
                f'Data must be identical for pairwise analysis, but length of files {system_datasets[0]} ({num0}) != {system_datasets[1]} ({num1})'
            )
        if (
            loaders[0].user_defined_features_configs
            != loaders[1].user_defined_features_configs
        ):
            raise ValueError(
                "User defined features must be the same for pairwise analysis."
            )

    # Setup metadata
    metadata = {
        "dataset_name": dataset,
        "sub_dataset_name": sub_dataset,
        "language": language,
        "task_name": task,
        "reload_stat": reload_stat,
        "conf_value": args.conf_value,
    }
    metadata.update(loaders[0].user_defined_metadata_configs)

    if metric_names is not None:
        metadata["metric_names"] = metric_names

    # Run analysis
    reports = []
    for loader, system_dataset, system_full_path in zip(loaders, system_datasets, system_outputs):

        metadata.update(loader.user_defined_metadata_configs)

        report = get_processor(task).process(metadata=metadata, sys_output=system_dataset)
        reports.append(report)

        # save report to `output_dir_reports`
        x_file_name = os.path.basename(system_full_path).split(".")[0]
        report.write_to_directory(output_dir_reports, f"{x_file_name}.json")

        # generate figures and save them into  `output_dir_figures`
        if not os.path.exists(f"{output_dir_figures}/{x_file_name}"):
            os.makedirs(f"{output_dir_figures}/{x_file_name}")
        draw_bar_chart_from_report(f"{output_dir_reports}/{x_file_name}.json", f"{output_dir_figures}/{x_file_name}")

    
    if len(system_outputs) == 1:  # individual system analysis
        reports[0].print_as_json()
    else:  # pairwise analysis
        compare_analysis = get_pairwise_performance_gap(
            reports[0].to_dict(), reports[1].to_dict()
        )
        print(json.dumps(compare_analysis, indent=4))


if __name__ == '__main__':
    main()
