"""Operations over tensors.

TODO(gneubig):
  Thes could probably be made easier through using Pandas dataframes or numpy arrays
"""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np

from explainaboard.utils.logging import get_logger


def aggregate_score_tensor(
    score_tensor: dict,
    systems_aggregation: Optional[str] = None,
    datasets_aggregation: Optional[str] = None,
    languages_aggregation: Optional[str] = None,
) -> dict:
    """This function aggregates score tensor based on specified parameters.

    It can be done along three dimensions: system, dataset and language
    TODO(Pengfei):
     (1) this function could be duplicated
     (2) the way to implement the aggregation should be generalized

    Args:
        score_tensor: The tensor to aggregate
        systems_aggregation: How to aggregate over systems, e.g. "minus"
        datasets_aggregation: How to aggregate over datasets, e.g. "average"
        languages_aggregation: How to aggregate over langauges, e.g. "average"

    Returns:
        A tensor with the scores aggregated
    """
    if datasets_aggregation is not None and languages_aggregation is None:
        languages_aggregation = datasets_aggregation

    # Regarding language aggregation
    score_tensor_aggregated_language: dict = {}

    if languages_aggregation == "average":
        for system_name, m_value in score_tensor.items():
            score_tensor_aggregated_language[system_name] = {}
            for dataset_name, d_value in score_tensor[system_name].items():
                score_info_template = copy.deepcopy(
                    list(score_tensor[system_name][dataset_name].values())[0]
                )

                score_tensor_aggregated_language[system_name][dataset_name] = {}
                aggregated_score = np.average(
                    [
                        score["value"]
                        for score in score_tensor[system_name][dataset_name].values()
                    ]
                )
                score_info_template["value"] = aggregated_score
                score_tensor_aggregated_language[system_name][dataset_name][
                    "all_languages"
                ] = score_info_template

    # Regarding dataset aggregation
    score_tensor_aggregated_dataset: dict = {}
    if datasets_aggregation == "average":
        for system_name, m_value in score_tensor.items():
            score_tensor_aggregated_dataset[system_name] = {}
            aggregated_score = 0.0
            score_info_template = {}
            for dataset_name, d_value in score_tensor[system_name].items():
                score_info_template = copy.deepcopy(
                    list(score_tensor[system_name][dataset_name].values())[0]
                )
                aggregated_score += score_tensor_aggregated_language[system_name][
                    dataset_name
                ]["all_languages"]["value"]
            aggregated_score /= len(score_tensor[system_name].items())
            score_info_template["value"] = aggregated_score
            score_tensor_aggregated_dataset[system_name]["all_datasets"] = {}
            score_tensor_aggregated_dataset[system_name]["all_datasets"][
                "all_languages"
            ] = score_info_template

    # Regarding system aggregation
    if datasets_aggregation is not None:
        score_tensor = score_tensor_aggregated_dataset
    elif languages_aggregation is not None:
        score_tensor = score_tensor_aggregated_language

    score_tensor_aggregated_system: dict = {}
    if systems_aggregation == "minus":
        if len(score_tensor) != 2:
            raise ValueError("the number of systems should two")
        sys1_name, sys2_name = list(score_tensor.keys())
        score_tensor_aggregated_system[f"{sys1_name} V.S {sys2_name}"] = {}
        sys1_datasets = score_tensor[sys1_name].keys()
        sys2_datasets = score_tensor[sys2_name].keys()
        common_datasets = list(set(sys1_datasets) & set(sys2_datasets))
        for common_dataset in common_datasets:
            sys1_languages = score_tensor[sys1_name][common_dataset]
            sys2_languages = score_tensor[sys2_name][common_dataset]
            common_languages = list(set(sys1_languages) & set(sys2_languages))
            score_tensor_aggregated_system[f"{sys1_name} V.S {sys2_name}"][
                common_dataset
            ] = {}

            for common_language in common_languages:
                score_tensor_aggregated_system[f"{sys1_name} V.S {sys2_name}"][
                    common_dataset
                ][common_language] = score_tensor[sys1_name][common_dataset][
                    common_language
                ]
                aggregated_score = (
                    score_tensor[sys1_name][common_dataset][common_language]["value"]
                    - score_tensor[sys2_name][common_dataset][common_language]["value"]
                )
                score_tensor_aggregated_system[f"{sys1_name} V.S {sys2_name}"][
                    common_dataset
                ][common_language]["value"] = aggregated_score

    elif systems_aggregation == "combine":
        raise NotImplementedError

    if systems_aggregation is not None:
        return score_tensor_aggregated_system
    elif datasets_aggregation is not None:
        return score_tensor_aggregated_dataset
    elif languages_aggregation is not None:
        return score_tensor_aggregated_language
    else:
        return score_tensor


def filter_score_tensor(
    score_tensor: dict,
    systems: Optional[list],
    datasets: Optional[list],
    languages: Optional[list],
) -> dict:
    """Remove elements of the tensor that don't match certain filters.

    Args:
        score_tensor: The tensor of scores
        systems: The list of systems to include, or `None` for all
        datasets: The list of datasets to include, or `None` for all
        languages: The list of languages to include, or `None` for all

    Returns:
        The filtered tensor
    """
    score_tensor_copy = copy.deepcopy(score_tensor)

    score_tensor_filter: dict = {}
    for system_name, m_value in score_tensor.items():
        if systems is not None and system_name not in systems:
            continue
        score_tensor_filter[system_name] = {}
        for dataset_name, d_value in score_tensor[system_name].items():
            if datasets is not None and dataset_name not in datasets:
                continue
            score_tensor_filter[system_name][dataset_name] = {}
            for language_name, l_value in score_tensor[system_name][
                dataset_name
            ].items():
                if languages is not None and language_name not in languages:
                    continue
                score_tensor_filter[system_name][dataset_name][
                    language_name
                ] = score_tensor_copy[system_name][dataset_name][language_name]
    return score_tensor_filter


def print_score_tensor(score_tensor: dict) -> None:
    """Print the score_tensor.

    For example,
     ----------------------------------------
    System: CL-mt5base, Dataset: xnli
    Language:       ar      bg      de      el      en      es      fr
    Accuracy:       0.679   0.714   0.721   0.722   0.768   0.738   0.721

    ----------------------------------------
    System: CL-mlpp15out1sum, Dataset: xnli
    Language:       ar      bg      de      el      en      es      fr
    Accuracy:       0.696   0.739   0.735   0.739   0.787   0.768   0.730

    ----------------------------------------
    System: CL-mlpp15out1sum, Dataset: marc
    Language:       de      en      es      fr      ja      zh
    Accuracy:       0.933   0.915   0.934   0.926   0.915   0.871

    Args:
        score_tensor: The tensor to print out.
    """
    get_logger("report").info(score_tensor.keys())
    for system_name, m_value in score_tensor.items():
        for dataset_name, d_value in score_tensor[system_name].items():
            info_printed = (
                f"----------------------------------------\nSystem: "
                f"{system_name}, Dataset: "
                f"{dataset_name} \n"
            )
            info_printed += (
                "Language:\t"
                + "\t".join(score_tensor[system_name][dataset_name].keys())
                + "\n"
            )
            metric_name = list(score_tensor[system_name][dataset_name].values())[0][
                "metric_name"
            ]
            info_printed += (
                f"{metric_name}:\t"
                + "\t".join(
                    [
                        "{:.3f}".format(score["value"])
                        for score in score_tensor[system_name][dataset_name].values()
                    ]
                )
                + "\n"
            )
            get_logger("report").info(info_printed)
