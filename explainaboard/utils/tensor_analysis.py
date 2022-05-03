import copy
from typing import Optional

import numpy as np

from explainaboard.utils.logging import get_logger


def aggregate_score_tensor(
    score_tensor: dict,
    models_aggregation: Optional[str] = None,
    datasets_aggregation: Optional[str] = None,
    languages_aggregation: Optional[str] = None,
):
    """
    This function aggregate score tensor based on specified parameters along
    three dimensions: model, dataset and language
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
    score_tensor_aggregated_language: dict = {}

    if languages_aggregation == "average":
        for model_name, m_value in score_tensor.items():
            score_tensor_aggregated_language[model_name] = {}
            for dataset_name, d_value in score_tensor[model_name].items():
                score_info_template = copy.deepcopy(
                    list(score_tensor[model_name][dataset_name].values())[0]
                )

                score_tensor_aggregated_language[model_name][dataset_name] = {}
                aggregated_score = np.average(
                    [
                        score["value"]
                        for score in score_tensor[model_name][dataset_name].values()
                    ]
                )
                score_info_template["value"] = aggregated_score
                score_tensor_aggregated_language[model_name][dataset_name][
                    "all_languages"
                ] = score_info_template

    # Regarding dataset aggregation
    score_tensor_aggregated_dataset: dict = {}
    if datasets_aggregation == "average":
        for model_name, m_value in score_tensor.items():
            score_tensor_aggregated_dataset[model_name] = {}
            aggregated_score = 0.0
            score_info_template = {}
            for dataset_name, d_value in score_tensor[model_name].items():
                score_info_template = copy.deepcopy(
                    list(score_tensor[model_name][dataset_name].values())[0]
                )
                aggregated_score += score_tensor_aggregated_language[model_name][
                    dataset_name
                ]["all_languages"]["value"]
            aggregated_score /= len(score_tensor[model_name].items())
            score_info_template["value"] = aggregated_score
            score_tensor_aggregated_dataset[model_name]["all_datasets"] = {}
            score_tensor_aggregated_dataset[model_name]["all_datasets"][
                "all_languages"
            ] = score_info_template

    # Regarding model aggregation
    if datasets_aggregation is not None:
        score_tensor = score_tensor_aggregated_dataset
    elif languages_aggregation is not None:
        score_tensor = score_tensor_aggregated_language

    score_tensor_aggregated_model: dict = {}
    if models_aggregation == "minus":
        if len(score_tensor) != 2:
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
            score_tensor_aggregated_model[f"{sys1_name} V.S {sys2_name}"][
                common_dataset
            ] = {}

            for common_language in common_languages:
                score_tensor_aggregated_model[f"{sys1_name} V.S {sys2_name}"][
                    common_dataset
                ][common_language] = score_tensor[sys1_name][common_dataset][
                    common_language
                ]
                aggregated_score = (
                    score_tensor[sys1_name][common_dataset][common_language]["value"]
                    - score_tensor[sys2_name][common_dataset][common_language]["value"]
                )
                score_tensor_aggregated_model[f"{sys1_name} V.S {sys2_name}"][
                    common_dataset
                ][common_language]["value"] = aggregated_score

    elif models_aggregation == "combine":
        raise NotImplementedError

    if models_aggregation is not None:
        return score_tensor_aggregated_model
    elif datasets_aggregation is not None:
        return score_tensor_aggregated_dataset
    elif languages_aggregation is not None:
        return score_tensor_aggregated_language
    else:
        return score_tensor


def filter_score_tensor(
    score_tensor: dict,
    models: Optional[list],
    datasets: Optional[list],
    languages: Optional[list],
):
    """
    filter score tensor based on given models, datasets and languages
    """
    score_tensor_copy = copy.deepcopy(score_tensor)

    score_tensor_filter: dict = {}
    for model_name, m_value in score_tensor.items():
        if models is not None and model_name not in models:
            continue
        score_tensor_filter[model_name] = {}
        for dataset_name, d_value in score_tensor[model_name].items():
            if datasets is not None and dataset_name not in datasets:
                continue
            score_tensor_filter[model_name][dataset_name] = {}
            for language_name, l_value in score_tensor[model_name][
                dataset_name
            ].items():
                if languages is not None and language_name not in languages:
                    continue
                score_tensor_filter[model_name][dataset_name][
                    language_name
                ] = score_tensor_copy[model_name][dataset_name][language_name]
    return score_tensor_filter


def print_score_tensor(score_tensor: dict):
    """
    print the score_tensor, for example,
     ----------------------------------------
    Model: CL-mt5base, Dataset: xnli
    Language:       ar      bg      de      el      en      es      fr
    Accuracy:       0.679   0.714   0.721   0.722   0.768   0.738   0.721

    ----------------------------------------
    Model: CL-mlpp15out1sum, Dataset: xnli
    Language:       ar      bg      de      el      en      es      fr
    Accuracy:       0.696   0.739   0.735   0.739   0.787   0.768   0.730

    ----------------------------------------
    Model: CL-mlpp15out1sum, Dataset: marc
    Language:       de      en      es      fr      ja      zh
    Accuracy:       0.933   0.915   0.934   0.926   0.915   0.871

    """
    get_logger('report').info(score_tensor.keys())
    for model_name, m_value in score_tensor.items():
        for dataset_name, d_value in score_tensor[model_name].items():
            info_printed = (
                f"----------------------------------------\nModel: "
                f"{model_name}, Dataset: "
                f"{dataset_name} \n"
            )
            info_printed += (
                "Language:\t"
                + "\t".join(score_tensor[model_name][dataset_name].keys())
                + "\n"
            )
            metric_name = list(score_tensor[model_name][dataset_name].values())[0][
                "metric_name"
            ]
            info_printed += (
                f"{metric_name}:\t"
                + "\t".join(
                    [
                        '{:.3f}'.format(score["value"])
                        for score in score_tensor[model_name][dataset_name].values()
                    ]
                )
                + "\n"
            )
            get_logger('report').info(info_printed)
