"""Meta-analyses regarding flipping of ranks between systems."""

from __future__ import annotations

import itertools

from explainaboard.info import SysOutputInfo
from explainaboard.meta_analyses.meta_analysis import MetaAnalysis
from explainaboard.meta_analyses.utils import report_to_sysout
from explainaboard.utils.typing_utils import unwrap


class RankFlippingMetaAnalysis(MetaAnalysis):
    """A class to perform meta-analysis of how ranks flip between buckets."""

    def __init__(self, model1_report: SysOutputInfo, model2_report: SysOutputInfo):
        """Constructor.

        Args:
            model1_report: The report for the first model.
            model2_report: The report for the second model.
        """
        self.model1_report = model1_report
        self.model2_report = model2_report

    def run_meta_analysis(self) -> dict | list:
        """Return the result of the meta-analysis."""
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
    def _is_flipped(val1: float, val2: float, reference: bool) -> bool:
        """Check whether values are flipped.

        This function checks whether `val1` and `val2` stands in the *opposite*
        relationship as what is expected in `reference`. This is a sort of signal
        for "surprise".

        Args:
            val1: the first value
            val2: the second value
            reference: true if we expect `val1` should be greater than `val2`.

        Returns:
            True if we are surprised by the result
        """
        return (val1 > val2) != reference

    def _metrics_to_metadata(self) -> dict:
        """Turns a `metric_configs` object into a metadata object for system output.

        Uses the metric configs of `self.model1_report` as metadata.

        Returns:
            The metadata dictionary.
        """
        metric_config_dict_iter = (
            level.metric_configs for level in self.model1_report.analysis_levels
        )
        # TODO(odashi): This iterator may generate name collision.
        metric_names_iter = itertools.chain.from_iterable(metric_config_dict_iter)

        metadata = {
            "custom_features": {
                name: {
                    "dtype": "string",  # for rank flipping, True or False
                    "description": name,
                    "num_buckets": 2,  # for rank flipping, True or False
                }
                for name in metric_names_iter
            }
        }
        return metadata

    def _get_reference_info(self, metadata: dict) -> dict:
        """Return whether model1 has a higher score than model2 for each metric.

        Interpretation: this tells us which model (model1 or model2) we should
        expect to outperform the other, for each metric.

        The rank-flipping meta-analysis will then reveal which buckets, and how
        many, subvert this expectation.
        """
        m1_overall = list(
            itertools.chain.from_iterable(unwrap(self.model1_report.results.overall))
        )
        m2_overall = list(
            itertools.chain.from_iterable(unwrap(self.model2_report.results.overall))
        )
        model1_metric_is_greater = {
            metric_name: m1_overall[metric_name].value > m2_overall[metric_name].value
            for metric_name in metadata["custom_features"].keys()
        }
        return model1_metric_is_greater

    def _get_aggregated_sysout(
        self,
        model1_buckets: list[dict],
        model2_buckets: list[dict],
        reference_dict: dict,
        metadata: dict,
    ) -> list[dict]:
        """Find the rank-flipping statistics for each bucket.

        Check whether, for each bucket,
        the relationship between model1's bucket-level metrics and model2's bucket-level
        metrics is the opposite from what is expected.

        Args:
            model1_buckets: The buckets for model 1
            model2_buckets: The buckets for model 2
            reference_dict: Whether we expect model 1 to be better than model 2
            metadata: Contains `custom_features`, the features to analyze

        Returns:
            A list of dictionaries indicating the rank flipping statistics.
        """
        paired_score_examples = []
        for m1_bucket, m2_bucket in zip(model1_buckets, model2_buckets):

            # bucket info (feature name, bucket interval, bucket name, bucket size)
            # should match exactly
            if m1_bucket["feature_name"] != m2_bucket["feature_name"]:
                raise ValueError(
                    f"feature name does not match:\n{m1_bucket} vs {m2_bucket}"
                )
            if m1_bucket["bucket_interval"] != m2_bucket["bucket_interval"]:
                raise ValueError(
                    f"bucket interval does not match:\n{m1_bucket} vs {m2_bucket}"
                )
            if m1_bucket["bucket_name"] != m2_bucket["bucket_name"]:
                raise ValueError(
                    f"bucket name does not match:\n{m1_bucket} vs {m2_bucket}"
                )
            if m1_bucket["bucket_size"] != m2_bucket["bucket_size"]:
                raise ValueError(
                    f"bucket size does not match:\n{m1_bucket} vs {m2_bucket}"
                )

            # calculate difference metrics
            example = {
                "feature_name": m1_bucket["feature_name"],
                "bucket_interval": m1_bucket["bucket_interval"],
                "bucket_name": m1_bucket["bucket_name"],
                "bucket_size": m1_bucket["bucket_size"],
            }
            for feature in metadata["custom_features"].keys():
                reference = reference_dict[feature]
                example[feature] = RankFlippingMetaAnalysis._is_flipped(
                    m1_bucket[feature], m2_bucket[feature], reference
                )
            paired_score_examples.append(example)

        return paired_score_examples

    def _bucket_aggregated_sysout(
        self, aggregated_sysout: list[dict], metadata: dict
    ) -> dict:
        """Taking paired_sysout do bucketing based on the metadata provided.

        Here we write the code which does the bucketing directly in this method,
        but in the future we may consider using the base ExplainaBoard package
        to do it for us, since `paired_sysout` is already in the format of a
        legitimate system output.

        Args:
            aggregated_sysout: The system output aggregated according to
              `_get_aggregated_sysout()`
            metadata: The metadata containing `custom_features` indicating which
              features to analyze.

        Returns:
            A dictionary with counts of how often the rank is flipped.
        """
        # for rank-flipping, it's very easy to bucket, since there are only 2
        # buckets (True or False). Let's just immediately calculate it here.
        rank_flipping_buckets = {
            metric_name: {"ranking_same": 0, "ranking_flipped": 0}
            for metric_name in metadata["custom_features"].keys()
        }

        for example in aggregated_sysout:
            for metric_name in metadata["custom_features"].keys():
                if example[metric_name]:
                    rank_flipping_buckets[metric_name]["ranking_flipped"] += 1
                else:
                    rank_flipping_buckets[metric_name]["ranking_same"] += 1
        return rank_flipping_buckets
