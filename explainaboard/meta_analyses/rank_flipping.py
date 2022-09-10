from __future__ import annotations

import itertools

from explainaboard.info import SysOutputInfo
from explainaboard.meta_analyses.utils import report_to_sysout
from explainaboard.utils.typing_utils import unwrap


class RankFlippingMetaAnalysis:  # (can inherit from an abstract MetaAnalysis class)
    def __init__(self, model1_report: SysOutputInfo, model2_report: SysOutputInfo):
        self.model1_report = model1_report
        self.model2_report = model2_report

    def run_meta_analysis(self):
        '''
        This method is what the user will call.
        '''

        # construct the new "metadata", treating each metric as a "feature"
        metadata = self._metrics_to_metadata()

        # construct the new "system outputs", treating each bucket from each
        # report as an "example/observation" to be analyzed/bucketed further
        model1_buckets = report_to_sysout(self.model1_report)
        model2_buckets = report_to_sysout(self.model2_report)

        # get reference info
        reference_dict = self._get_reference_info(metadata)
        self.reference_dict = reference_dict

        # calculate paired metrics. This is what will be bucketed further to
        # obtain the results of meta-analysis.
        aggregated_sysout = self._get_aggregated_sysout(
            model1_buckets, model2_buckets, reference_dict, metadata
        )
        self.aggregated_sysout = (
            aggregated_sysout  # save before bucketing for more fine-grained analysis
        )

        # bucket the paired system outputs
        return self._bucket_aggregated_sysout(aggregated_sysout, metadata)

    @staticmethod
    def is_flipped(val1: float, val2: float, reference: bool) -> bool:
        '''
        `reference` contains our expectation/baseline of whether `val1` should be
        greater than `val2`.

        This function checks whether `val1` and `val2` stands in the *opposite*
        relationship as what is expected in `reference`. This is a sort of signal
        for "surprise".
        '''
        return (val1 > val2) != reference

    def _metrics_to_metadata(self) -> dict:
        '''
        Turns a `metric_configs` object into a metadata object suitable
        for use as metadata in a system output file.

        Uses the metric configs of `self.model1_report` as metadata.
        '''
        metadata = {
            'custom_features': {
                metric_config.name: {
                    "dtype": "string",  # for rank flipping, True or False
                    "description": metric_config.name,
                    "num_buckets": 2,  # for rank flipping, True or False
                }
                for metric_config in itertools.chain.from_iterable(
                    [
                        x.metric_configs
                        for x in unwrap(self.model1_report.analysis_levels)
                    ]
                )
            }
        }
        return metadata

    def _get_reference_info(self, metadata: dict) -> dict:
        '''
        Returns a dictionary indicating, for each metric, whether model1's value
        for that metric is higher than model2's value.

        Interpretation: this tells us which model (model1 or model2) we should
        expect to outperform the other, for each metric.

        The rank-flipping meta-analysis will then reveal which buckets, and how
        many, subvert this expectation.
        '''
        m1_overall = list(
            itertools.chain.from_iterable(unwrap(self.model1_report.results.overall))
        )
        m2_overall = list(
            itertools.chain.from_iterable(unwrap(self.model2_report.results.overall))
        )
        model1_metric_is_greater = {
            metric_name: m1_overall[metric_name].value > m2_overall[metric_name].value
            for metric_name in metadata['custom_features'].keys()
        }
        return model1_metric_is_greater

    def _get_aggregated_sysout(
        self,
        model1_buckets: list[dict],
        model2_buckets: list[dict],
        reference_dict: dict,
        metadata: dict,
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

        paired_score_examples = []
        for m1_bucket, m2_bucket in zip(model1_buckets, model2_buckets):

            # bucket info (feature name, bucket interval, bucket name, bucket size)
            # should match exactly
            if m1_bucket['feature_name'] != m2_bucket['feature_name']:
                raise ValueError(
                    f'feature name does not match:\n{m1_bucket} vs {m2_bucket}'
                )
            if m1_bucket['bucket_interval'] != m2_bucket['bucket_interval']:
                raise ValueError(
                    f'bucket interval does not match:\n{m1_bucket} vs {m2_bucket}'
                )
            if m1_bucket['bucket_name'] != m2_bucket['bucket_name']:
                raise ValueError(
                    f'bucket name does not match:\n{m1_bucket} vs {m2_bucket}'
                )
            if m1_bucket['bucket_size'] != m2_bucket['bucket_size']:
                raise ValueError(
                    f'bucket size does not match:\n{m1_bucket} vs {m2_bucket}'
                )

            # calculate difference metrics
            example = {
                'feature_name': m1_bucket['feature_name'],
                'bucket_interval': m1_bucket['bucket_interval'],
                'bucket_name': m1_bucket['bucket_name'],
                'bucket_size': m1_bucket['bucket_size'],
            }
            for feature in metadata['custom_features'].keys():
                reference = reference_dict[feature]
                example[feature] = RankFlippingMetaAnalysis.is_flipped(
                    m1_bucket[feature], m2_bucket[feature], reference
                )
            paired_score_examples.append(example)

        return paired_score_examples

    def _bucket_aggregated_sysout(
        self, aggregated_sysout: list[dict], metadata: dict
    ) -> dict:
        '''
        Taking paired_sysout as a system output, do bucketing on it based on
        the metadata provided.

        Here we write the code which does the bucketing directly in this method,
        but in the future we may consider using the base ExplainaBoard package
        to do it for us, since `paired_sysout` is already in the format of a
        legitimate system output.
        '''

        # for rank-flipping, it's very easy to bucket, since there are only 2
        # buckets (True or False). Let's just immediately calculate it here.
        rank_flipping_buckets = {
            metric_name: {'ranking_same': 0, 'ranking_flipped': 0}
            for metric_name in metadata['custom_features'].keys()
        }

        for example in aggregated_sysout:
            for metric_name in metadata['custom_features'].keys():
                if example[metric_name] is True:
                    rank_flipping_buckets[metric_name]['ranking_flipped'] += 1
                else:
                    rank_flipping_buckets[metric_name]['ranking_same'] += 1
        return rank_flipping_buckets
