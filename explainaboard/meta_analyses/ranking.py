"""A class for meta-analysis of rankings."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd

from explainaboard.info import SysOutputInfo
from explainaboard.meta_analyses.meta_analysis import MetaAnalysis
from explainaboard.meta_analyses.utils import report_to_sysout
from explainaboard.utils.typing_utils import unwrap


class RankingMetaAnalysis(MetaAnalysis):
    """A class for meta-analysis of rankings."""

    def __init__(self, model_reports: dict[str, SysOutputInfo]):
        """Initialize the meta-analysis with model reports."""
        self.model_names: list[str] = list(model_reports.keys())
        self.num_models: int = len(self.model_names)
        self.model_ids: list[int] = list(
            np.arange(self.num_models)
        )  # (0, ..., n_models-1)
        self.model_reports: list[SysOutputInfo] = list(model_reports.values())

    def run_meta_analysis(self) -> dict | list:
        """Run a meta analysis over the ranking over different buckets."""
        # construct the new "metadata", treating each metric as a "feature"
        metadata = self._metrics_to_metadata()
        self.metadata = metadata

        # construct the new "system outputs", treating each bucket from each
        # report as an "example/observation" to be bucketed further
        buckets_per_model: list[list[dict]] = []
        for model_report in self.model_reports:
            model_buckets: list[dict] = report_to_sysout(model_report)
            buckets_per_model.append(model_buckets)

        # overall performance comparison (on dataset-level), which will be a
        # useful reference/baseline when analyzing bucket-level performance
        reference_dict = self._get_reference_info(metadata)
        self.reference_dict = reference_dict

        # aggregates buckets from all models, and returns a value as a function
        # of all the models.
        # In this case, return the ranking of each model (relative to each other)
        # for each bucket and each metric
        aggregated_sysout = self._get_aggregated_sysout(
            buckets_per_model, reference_dict, metadata
        )
        self.aggregated_sysout = aggregated_sysout

        return aggregated_sysout

    def get_ranking_table(self, metric: str) -> pd.DataFrame:
        """Returns the ranking table for a given metric as a pandas DataFrame.

        Args:
            metric: The name of the metric for which to calculate.

        Returns:
            The ranking table.
        """
        if metric not in self.feature_names:
            raise ValueError(f"metric {metric} does not exist in original model report")
        metric_id = self.feature_names.index(metric)
        metric_ranking_table = self.ranking_table[metric_id]
        metric_ranking_df = pd.DataFrame(
            metric_ranking_table + 1, columns=self.bucket_names, index=self.model_names
        )
        return metric_ranking_df

    @staticmethod
    def _get_ranks_of(values: list[float], ids: list[int] | list[str]) -> list[int]:
        """Returns the rank of each element in `ids`.

        Rankings are high-to-low and zero-indexed.

        Args:
            values: The values to use for the ranking.
            ids: The IDs over which to perform ranking.

        Returns:
            A list of the ranks.
        """
        values_np = np.array(values)
        sort_idx = (np.argsort(values_np)[::-1]).tolist()
        return [sort_idx.index(i) for i in ids]

    def _metrics_to_metadata(self) -> dict:
        """Turns a `metric_configs` object into a metadata object.

        This metadata object is suitable for use as metadata in a system output file.

        Uses the metric configs of `self.model1_report` as metadata.
        """
        model1 = self.model_reports[0]

        metric_config_dict_iter = (
            level.metric_configs for level in model1.analysis_levels
        )
        # TODO(odashi): This iterator may generate name collision.
        metric_names_iter = itertools.chain.from_iterable(metric_config_dict_iter)

        metadata = {
            "custom_features": {
                name: {
                    "dtype": "string",
                    "description": name,
                    "num_buckets": len(self.model_reports),
                }
                for name in metric_names_iter
            }
        }
        return metadata

    def _get_reference_info(self, metadata: dict) -> dict:
        """Return ranks of each bucket for each feature."""
        model_overall_results = [
            list(itertools.chain.from_iterable(unwrap(r.results.overall)))
            for r in self.model_reports
        ]
        reference_info: dict[str, Any] = {
            "feature_name": "overall",
            "bucket_interval": None,
            "bucket_name": "",
            "bucket_size": -1,
        }
        for feature_id, feature in enumerate(metadata["custom_features"].keys()):
            values = [
                model_result[feature].value for model_result in model_overall_results
            ]
            model_ranks = RankingMetaAnalysis._get_ranks_of(values, self.model_ids)
            reference_info[f"{feature}_model_ranking"] = model_ranks

        return reference_info

    def _get_aggregated_sysout(
        self, model_sysouts: list[list[dict]], reference_dict: dict, metadata: dict
    ) -> list[dict]:
        """Aggregate multiple system outputs into one."""
        # check that each model's meta-level system outputs has the same length
        num_buckets = len(model_sysouts[0])
        for model_sysout in model_sysouts:
            if len(model_sysout) != num_buckets:
                raise ValueError("length of model's meta-sysouts do not match")

        aggregated_examples = []
        ranking_table = np.zeros(
            (
                len(metadata["custom_features"].keys()),  # 'Hits1', 'Hits2', ...
                len(model_sysouts),  # 'rotate', 'rescal', ...
                num_buckets + 1,  # 'feature1_(a,b)', 'feature1_(c,d)', ..., 'overall'
            ),
            dtype=int,
        )
        bucket_names = []
        for bucket_id in range(num_buckets):

            buckets_per_model = [
                model_sysout[bucket_id] for model_sysout in model_sysouts
            ]
            bucket_info = buckets_per_model[0]
            bucket_names.append(
                f'{bucket_info["feature_name"]}_{bucket_info["bucket_name"]}'
            )

            # calculate difference metrics
            example = {
                "feature_name": bucket_info["feature_name"],
                "bucket_interval": bucket_info["bucket_interval"],
                "bucket_name": bucket_info["bucket_name"],
                "bucket_size": bucket_info["bucket_size"],
            }
            for feature_id, feature in enumerate(metadata["custom_features"].keys()):
                values = [model_bucket[feature] for model_bucket in buckets_per_model]
                model_ranks = RankingMetaAnalysis._get_ranks_of(values, self.model_ids)
                example[f"{feature}_model_ranking"] = model_ranks
                ranking_table[feature_id, :, bucket_id] = model_ranks
            aggregated_examples.append(example)

        # save overall results into last column of table
        for feature_id, feature in enumerate(metadata["custom_features"].keys()):
            overall_model_ranks = reference_dict[f"{feature}_model_ranking"]
            ranking_table[feature_id, :, -1] = overall_model_ranks

        self.ranking_table = ranking_table
        self.feature_names = list(
            metadata["custom_features"].keys()
        )  # 'Hits1', 'Hits2', ...
        self.bucket_names = bucket_names + ["overall"]

        return aggregated_examples
