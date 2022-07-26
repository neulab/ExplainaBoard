from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd

from explainaboard.info import SysOutputInfo
from explainaboard.meta_analyses.utils import report_to_sysout
from explainaboard.utils.typing_utils import unwrap


class RankingMetaAnalysis:  # (can inherit from an abstract MetaAnalysis class)
    def __init__(self, model_reports: dict[str, SysOutputInfo]):
        self.model_names = list(model_reports.keys())
        self.num_models = len(self.model_names)
        self.model_ids = np.arange(self.num_models).tolist()  # (0, ..., n_models-1)
        self.model_reports = list(model_reports.values())

    def run_meta_analysis(self):
        '''
        This method is what the user will call.
        '''

        # construct the new "metadata", treating each metric as a "feature"
        metadata = self._metrics_to_metadata()
        self.metadata = metadata

        # construct the new "system outputs", treating each bucket from each
        # report as an "example/observation" to be bucketed further
        buckets_per_model = []
        for model_report in self.model_reports:
            model_buckets = report_to_sysout(model_report)
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

    def get_ranking_table(self, metric: str):
        '''
        Returns the ranking table for a given metric as a pandas DataFrame.
        Ranking is 1-indexed.
        '''
        if metric not in self.feature_names:
            raise ValueError(f'metric {metric} does not exist in original model report')
        metric_id = self.feature_names.index(metric)
        metric_ranking_table = self.ranking_table[metric_id]
        metric_ranking_df = pd.DataFrame(
            metric_ranking_table + 1, columns=self.bucket_names, index=self.model_names
        )
        return metric_ranking_df

    @staticmethod
    def _get_ranks_of(values: list[float], ids: list[Union[int, str]]) -> list[int]:
        '''
        Returns the rank of each element of `ids` based on `values`, high-to-low.
        0-indexed.
        '''
        values = np.array(values)
        sort_idx = (np.argsort(values)[::-1]).tolist()
        return [sort_idx.index(i) for i in ids]

    def _metrics_to_metadata(self) -> dict:
        '''
        Turns a `metric_configs` object into a metadata object suitable
        for use as metadata in a system output file.

        Uses the metric configs of `self.model1_report` as metadata.
        '''
        model1 = self.model_reports[0]
        metadata = {
            'custom_features': {
                metric_config.name: {
                    "dtype": "string",
                    "description": metric_config.name,
                    "num_buckets": len(self.model_reports),
                }
                for metric_config in unwrap(model1.metric_configs)
            }
        }
        return metadata

    def _get_reference_info(self, metadata: dict) -> dict:
        '''
        Returns a dictionary indicating, for each metric, whether model1's value
        for that metric is higher than model2's value. Here, as in most cases,
        we use the overall results to establish this expectation.

        Interpretation: this tells us which model (model1 or model2) we should
        expect to outperform the other, for each metric.

        The rank-flipping meta-analysis will then reveal which buckets, and how
        many, subvert this expectation.
        '''
        model_overall_results = [unwrap(r.results.overall) for r in self.model_reports]
        reference_info = {
            'feature_name': 'overall',
            'bucket_interval': '',
            'bucket_size': -1,
        }
        for feature_id, feature in enumerate(metadata['custom_features'].keys()):
            values = [
                model_result[feature].value for model_result in model_overall_results
            ]
            model_ranks = RankingMetaAnalysis._get_ranks_of(values, self.model_ids)
            reference_info[f'{feature}_model_ranking'] = model_ranks

        return reference_info

    def _get_aggregated_sysout(
        self, model_sysouts: list[dict], reference_dict: dict, metadata: dict
    ) -> list[dict]:
        '''
        Many meta-analyses require a quantity which is based on the relationship
        between the metrics of the two models we want to compare.

        For example, in rank-flipping, we are interested in whether, for each bucket,
        the relationship between model1's bucket-level metrics and model2's bucket-level
        metrics is the opposite from what is expected.

        The expectation between the relationship between model1 and model2 is passed in
        through `reference_dict`.
        '''

        # check that each model's meta-level system outputs has the same length
        num_buckets = len(model_sysouts[0])
        for model_sysout in model_sysouts:
            if len(model_sysout) != num_buckets:
                raise ValueError('length of model\'s meta-sysouts do not match')

        aggregated_examples = []
        ranking_table = np.zeros(
            (
                len(metadata['custom_features'].keys()),  # 'Hits1', 'Hits2', ...
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
                f'{bucket_info["feature_name"]}_{bucket_info["bucket_interval"]}'
            )

            # calculate difference metrics
            example = {
                'feature_name': bucket_info['feature_name'],
                'bucket_interval': bucket_info['bucket_interval'],
                'bucket_size': bucket_info['bucket_size'],
            }
            for feature_id, feature in enumerate(metadata['custom_features'].keys()):
                values = [model_bucket[feature] for model_bucket in buckets_per_model]
                model_ranks = RankingMetaAnalysis._get_ranks_of(values, self.model_ids)
                example[f'{feature}_model_ranking'] = model_ranks
                ranking_table[feature_id, :, bucket_id] = model_ranks
            aggregated_examples.append(example)

        # save overall results into last column of table
        for feature_id, feature in enumerate(metadata['custom_features'].keys()):
            overall_model_ranks = reference_dict[f'{feature}_model_ranking']
            ranking_table[feature_id, :, -1] = overall_model_ranks

        self.ranking_table = ranking_table
        self.feature_names = list(
            metadata['custom_features'].keys()
        )  # 'Hits1', 'Hits2', ...
        self.bucket_names = bucket_names + ['overall']

        return aggregated_examples
