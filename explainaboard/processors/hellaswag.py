from typing import List, Dict, Any

import explainaboard.metric
from explainaboard import feature
from explainaboard.info import BucketPerformance, SysOutputInfo
from explainaboard.processors.processor import Processor
from explainaboard.processors.processor_registry import register_processor
from explainaboard.tasks import TaskType
from explainaboard.utils.feature_funcs import get_similarity_by_sacrebleu
from explainaboard.utils.py_utils import sort_dict


@register_processor(TaskType.hellaswag)
class HellaswagProcessor(Processor):
    _task_type = TaskType.hellaswag
    _features = feature.Features(
        {
            "ctx_a": feature.Value("string", is_bucket=False),
            "ctx_b": feature.Value("string", is_bucket=False),
            "ctx": feature.Value("string", is_bucket=False),
            "endings": feature.Sequence(feature.Value("string")),
            "true_label": feature.ClassLabel(names=["1", "0"], is_bucket=False),
            "predicted_label": feature.ClassLabel(names=["1", "0"], is_bucket=False),
            "ind": feature.Value(
                dtype="float",
                is_bucket=True,
                bucket_info=feature.BucketInfo(
                    method="bucket_attribute_specified_bucket_value",
                    number=4,
                    setting=(),
                ),
            ),
            "activity_label": feature.Value(
                "string",
                is_bucket=True,
                bucket_info=feature.BucketInfo(
                    method="bucket_attribute_discrete_value", number=10, setting=1
                ),
            ),
            "ctx_length": feature.Value(
                dtype="float",
                is_bucket=True,
                bucket_info=feature.BucketInfo(
                    method="bucket_attribute_specified_bucket_value",
                    number=4,
                    setting=(),
                ),
            ),
            "ctx_a_length_divided_b": feature.Value(
                dtype="float",
                is_bucket=True,
                bucket_info=feature.BucketInfo(
                    method="bucket_attribute_specified_bucket_value",
                    number=4,
                    setting=(),
                ),
            ),
            "true_answer_length": feature.Value(
                dtype="float",
                is_bucket=True,
                bucket_info=feature.BucketInfo(
                    method="bucket_attribute_specified_bucket_value",
                    number=4,
                    setting=(),
                ),
            ),
            "similarity_ctx_true_answer": feature.Value(
                dtype="float",
                is_bucket=True,
                bucket_info=feature.BucketInfo(
                    method="bucket_attribute_specified_bucket_value",
                    number=4,
                    setting=(),
                ),
            ),
        }
    )
    _default_metrics = ["Accuracy"]

    # --- Feature functions accessible by ExplainaboardBuilder._get_feature_func()
    def _get_similarity_ctx_true_answer(self, existing_features: dict):
        true_label = int(existing_features["true_label"])
        true_answer = existing_features["endings"][true_label]

        return get_similarity_by_sacrebleu(existing_features["ctx"], true_answer)

    # define function for incomplete features
    def _get_ctx_length(self, existing_features: dict):
        return len(self._tokenizer(existing_features["ctx"]))

    # define function for incomplete features
    def _get_ctx_a_length_divided_b(self, existing_features: dict):
        return (
            len(self._tokenizer(existing_features["ctx_a"]))
            * 1.0
            / len(self._tokenizer(existing_features["ctx_b"]))
        )

    def _get_true_answer_length(self, existing_features: dict):
        true_label = int(existing_features["true_label"])
        true_answer = existing_features["endings"][true_label]
        return len(self._tokenizer(true_answer))

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
        scoring_stats: Any = None,
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
