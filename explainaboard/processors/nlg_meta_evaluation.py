from __future__ import annotations

from explainaboard import feature, TaskType
from explainaboard.info import SysOutputInfo
from explainaboard.metrics.metric import MetricConfig
from explainaboard.metrics.nlg_meta_evaluation import (
    KtauCorrelationConfig,
    PearsonCorrelationConfig,
)
from explainaboard.processors.processor import Processor
from explainaboard.processors.processor_registry import register_processor
from explainaboard.utils.feature_funcs import get_basic_words, get_lexical_richness
from explainaboard.utils.typing_utils import unwrap


@register_processor(TaskType.nlg_meta_evaluation)
class NLGMetaEvaluationProcessor(Processor):
    @classmethod
    def task_type(cls) -> TaskType:
        return TaskType.nlg_meta_evaluation

    @classmethod
    def default_features(cls) -> feature.Features:
        return feature.Features(
            {
                "sys_name": feature.Value("string"),
                "seg_id": feature.Value("string"),
                "test_set": feature.Value("string"),
                "src": feature.Value("string"),
                "ref": feature.Value("string"),
                "sys": feature.Value("string"),
                "manual_raw": feature.Value("string"),
                "manual_z": feature.Value("string"),
                "auto_score": feature.Value("string"),
                "mean_ref_sys_length": feature.Value(
                    dtype="float",
                    description="text length in tokens",
                    is_bucket=True,
                    bucket_info=feature.BucketInfo(
                        method="bucket_attribute_specified_bucket_value",
                        number=4,
                        setting=(),
                    ),
                ),
                "minus_ref_sys_length": feature.Value(
                    dtype="float",
                    description="number of ref tokens minus number of sys tokens",
                    is_bucket=True,
                    bucket_info=feature.BucketInfo(
                        method="bucket_attribute_specified_bucket_value",
                        number=4,
                        setting=(),
                    ),
                ),
                "manual_score": feature.Value(
                    dtype="float",
                    description="manual score of an example",
                    is_bucket=True,
                    bucket_info=feature.BucketInfo(
                        method="bucket_attribute_specified_bucket_value",
                        number=4,
                        setting=(),
                    ),
                ),
                "basic_words": feature.Value(
                    dtype="float",
                    description="the ratio of basic words",
                    is_bucket=True,
                    bucket_info=feature.BucketInfo(
                        method="bucket_attribute_specified_bucket_value",
                        number=4,
                        setting=(),
                    ),
                ),
                "lexical_richness": feature.Value(
                    dtype="float",
                    description="lexical diversity",
                    is_bucket=True,
                    bucket_info=feature.BucketInfo(
                        method="bucket_attribute_specified_bucket_value",
                        number=4,
                        setting=(),
                    ),
                ),
            }
        )

    @classmethod
    def default_metrics(
        cls, source_language=None, target_language=None
    ) -> list[MetricConfig]:
        return [
            KtauCorrelationConfig(name='SegKtauCorr', group_by='segment'),
            PearsonCorrelationConfig(name='SysPearsonCorr', group_by='system'),
        ]

    # --- Feature functions accessible by ExplainaboardBuilder._get_feature_func()
    def _get_mean_ref_sys_length(
        self, sys_info: SysOutputInfo, existing_features: dict
    ):
        return (
            len(unwrap(sys_info.target_tokenizer)(existing_features["ref"]))
            + len(unwrap(sys_info.target_tokenizer)(existing_features["sys"]))
        ) / 2

    def _get_minus_ref_sys_length(
        self, sys_info: SysOutputInfo, existing_features: dict
    ):
        return len(unwrap(sys_info.target_tokenizer)(existing_features["ref"])) - len(
            unwrap(sys_info.target_tokenizer)(existing_features["sys"])
        )

    def _get_manual_score(self, sys_info: SysOutputInfo, existing_features: dict):
        return (
            float(existing_features["manual_z"])
            if existing_features["manual_z"] != ''
            else 0
        )

    def _get_basic_words(self, sys_info: SysOutputInfo, existing_feature: dict):
        return get_basic_words(existing_feature["ref"])

    def _get_lexical_richness(self, sys_info: SysOutputInfo, existing_feature: dict):
        return get_lexical_richness(existing_feature["ref"])

    def _get_true_label(self, data_point: dict):
        """
        Get the true label from a data point. Returns "true_label" by default, but can
        be overloaded.
        :param data_point: the data point under consideration
        :return: the true label for the output
        """
        return [
            data_point["sys_name"],
            data_point["seg_id"],
            data_point["manual_raw"],
            data_point["manual_z"],
        ]

    def _get_predicted_label(self, data_point: dict):
        """
        Get the predicted label from a data point. Returns "predicted_label" by default,
        but can be overloaded.
        :param data_point: the data point under consideration
        :return: the predicted label for the output
        """
        return data_point["auto_score"]
