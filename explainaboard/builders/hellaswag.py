from typing import List, Dict
from explainaboard.builders import ExplainaboardBuilder
from explainaboard.info import BucketPerformance, SysOutputInfo
from explainaboard.utils.feature_funcs import get_similarity_by_sacrebleu
import explainaboard.metric
from explainaboard.utils.py_utils import sort_dict

"""TODO
"""


class HellaswagExplainaboardBuilder(ExplainaboardBuilder):
    """
    Input: System Output file List[dict];  Metadata info
    Output: Analysis
    """

    def __init__(self):
        super().__init__()

    # --- Feature functions accessible by ExplainaboardBuilder._get_feature_func()
    def _get_similarity_ctx_true_answer(self, existing_features: dict):
        true_label = int(existing_features["true_label"])
        true_answer = existing_features["endings"][true_label]

        return get_similarity_by_sacrebleu(existing_features["ctx"], true_answer)

    # define function for incomplete features
    def _get_ctx_length(self, existing_features: dict):
        return len(existing_features["ctx"].split(" "))

    # define function for incomplete features
    def _get_ctx_a_length_divided_b(self, existing_features: dict):
        return (
            len(existing_features["ctx_a"].split(" "))
            * 1.0
            / len(existing_features["ctx_b"].split(" "))
        )

    def _get_true_answer_length(self, existing_features: dict):
        true_label = int(existing_features["true_label"])
        true_answer = existing_features["endings"][true_label]
        return len(true_answer.split(" "))

    def _get_activity_label(self, existing_feature: dict):
        return str(existing_feature["activity_label"])

    def _get_ind(self, existing_feature: dict):
        return float(existing_feature["ind"])

    # --- End feature functions

    # TODO(gneubig): this should be generalized
    def get_bucket_performance(
        self,
        sys_info: SysOutputInfo,
        sys_output: List[dict],
        samples_over_bucket: Dict[str, List[int]],
    ) -> Dict[str, List[BucketPerformance]]:
        """
        This function defines how to get bucket-level performance w.r.t a given feature (e.g., sentence length)
        :param sys_info: Information about the system output
        :param sys_output: The system output itself
        :param samples_over_bucket: a dictionary mapping bucket interval names to sample IDs for that bucket
        :return: bucket_name_to_performance: a dictionary that maps bucket names to bucket performance
        """

        bucket_name_to_performance = {}
        for bucket_interval, sample_ids in samples_over_bucket.items():

            bucket_true_labels = []
            bucket_predicted_labels = []
            bucket_cases = []

            for sample_id in sample_ids:

                true_label = sys_output[int(sample_id)]["true_label"]
                predicted_label = sys_output[int(sample_id)]["predicted_label"]
                s_id = sys_output[int(sample_id)]["id"]

                # get a bucket of true/predicted labels
                bucket_true_labels.append(true_label)
                bucket_predicted_labels.append(predicted_label)
                # get a bucket of cases (e.g., errors)
                if sys_info.is_print_case:
                    if true_label != predicted_label:
                        bucket_case = str(s_id)
                        bucket_cases.append(bucket_case)

            bucket_name_to_performance[bucket_interval] = []
            for metric_name in sys_info.metric_names:
                metric_func = getattr(explainaboard.metric, metric_name)
                one_metric = metric_func(
                    true_labels=bucket_true_labels,
                    predicted_labels=bucket_predicted_labels,
                    is_print_confidence_interval=sys_info.is_print_confidence_interval,
                )
                bucket_value_json = one_metric.evaluate()

                bucket_value = bucket_value_json["value"]
                confidence_score_low = bucket_value_json["confidence_score_low"]
                confidence_score_high = bucket_value_json["confidence_score_high"]

                # print(f"name:\t {one_metric._name} \n"
                #       f"value:\t {bucket_value}\n"
                #       f"confidence low\t {confidence_score_low}\n"
                #       f"confidence up \t {confidence_score_high}\n"
                #       f"---------------------------------")

                bucket_performance = BucketPerformance(
                    bucket_name=bucket_interval,
                    metric_name=metric_name,
                    value=bucket_value,
                    confidence_score_low=confidence_score_low,
                    confidence_score_high=confidence_score_high,
                    n_samples=len(bucket_true_labels),
                    bucket_samples=bucket_cases,
                )

                bucket_name_to_performance[bucket_interval].append(bucket_performance)

        return sort_dict(bucket_name_to_performance)
