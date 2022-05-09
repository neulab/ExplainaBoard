from __future__ import annotations

from typing import Any

from datalabs import aggregating

from explainaboard import feature, TaskType
from explainaboard.info import SysOutputInfo
from explainaboard.metric import AccuracyConfig, MetricConfig
from explainaboard.processors.processor import Processor
from explainaboard.processors.processor_registry import register_processor
from explainaboard.utils.feature_funcs import (
    accumulate_vocab_from_samples,
    feat_freq_rank,
    feat_num_oov,
    get_similarity_by_sacrebleu,
)
from explainaboard.utils.typing_utils import unwrap


@register_processor(TaskType.text_pair_classification)
class TextPairClassificationProcessor(Processor):
    @classmethod
    def task_type(cls) -> TaskType:
        return TaskType.text_classification

    @classmethod
    def default_features(cls) -> feature.Features:
        return feature.Features(
            {
                "text": feature.Value("string"),
                "true_label": feature.Value("string"),
                "predicted_label": feature.Value("string"),
                "label": feature.Value(
                    dtype="string",
                    is_bucket=True,
                    bucket_info=feature.BucketInfo(
                        method="bucket_attribute_discrete_value", number=4, setting=1
                    ),
                ),
                "text1_length": feature.Value(
                    dtype="float",
                    description="text1 length",
                    is_bucket=True,
                    bucket_info=feature.BucketInfo(
                        method="bucket_attribute_specified_bucket_value",
                        number=4,
                        setting=(),
                    ),
                ),
                "text2_length": feature.Value(
                    dtype="float",
                    description="text2 length",
                    is_bucket=True,
                    bucket_info=feature.BucketInfo(
                        method="bucket_attribute_specified_bucket_value",
                        number=4,
                        setting=(),
                    ),
                ),
                "similarity": feature.Value(
                    dtype="float",
                    description="two texts' similarity",
                    is_bucket=True,
                    bucket_info=feature.BucketInfo(
                        method="bucket_attribute_specified_bucket_value",
                        number=4,
                        setting=(),
                    ),
                ),
                "text1_divided_text2": feature.Value(
                    dtype="float",
                    description="diff of two texts' length",
                    is_bucket=True,
                    bucket_info=feature.BucketInfo(
                        method="bucket_attribute_specified_bucket_value",
                        number=4,
                        setting=(),
                    ),
                ),
                "num_oov": feature.Value(
                    dtype="float",
                    description="the number of out-of-vocabulary words",
                    is_bucket=True,
                    bucket_info=feature.BucketInfo(
                        method="bucket_attribute_specified_bucket_value",
                        number=4,
                        setting=(),
                    ),
                    require_training_set=True,
                ),
                "fre_rank": feature.Value(
                    dtype="float",
                    description=(
                        "the average rank of each word based on its frequency in "
                        "training set"
                    ),
                    is_bucket=True,
                    bucket_info=feature.BucketInfo(
                        method="bucket_attribute_specified_bucket_value",
                        number=4,
                        setting=(),
                    ),
                    require_training_set=True,
                ),
            }
        )

    @classmethod
    def default_metrics(
        cls, source_language=None, target_language=None
    ) -> list[MetricConfig]:
        return [AccuracyConfig(name='Accuracy')]

    @aggregating()
    def _statistics_func(self, samples, sys_info: SysOutputInfo):
        return {
            'source_vocab': accumulate_vocab_from_samples(
                samples, lambda x: x['text1'], unwrap(sys_info.source_tokenizer)
            ),
            'target_vocab': accumulate_vocab_from_samples(
                samples, lambda x: x['text2'], unwrap(sys_info.target_tokenizer)
            ),
        }

    # --- Feature functions accessible by ExplainaboardBuilder._get_feature_func()
    def _get_similarity(self, sys_info: SysOutputInfo, existing_features: dict):
        return get_similarity_by_sacrebleu(
            existing_features["text1"], existing_features["text2"]
        )

    def _get_text1_length(self, sys_info: SysOutputInfo, existing_features: dict):
        return len(unwrap(sys_info.source_tokenizer)(existing_features["text1"]))

    def _get_text2_length(self, sys_info: SysOutputInfo, existing_feature: dict):
        return len(unwrap(sys_info.target_tokenizer)(existing_feature["text2"]))

    def _get_text1_divided_text2(self, sys_info: SysOutputInfo, existing_feature: dict):
        return (
            len(unwrap(sys_info.source_tokenizer)(existing_feature["text1"]))
            * 1.0
            / len(unwrap(sys_info.target_tokenizer)(existing_feature["text2"]))
        )

    def _get_label(self, sys_info: SysOutputInfo, existing_feature: dict):
        return existing_feature["true_label"]

    # training set dependent features
    def _get_num_oov(
        self, sys_info: SysOutputInfo, existing_features: dict, statistics: Any
    ):
        return feat_num_oov(
            existing_features,
            statistics['source_vocab'],
            lambda x: x['text1'],
            unwrap(sys_info.source_tokenizer),
        ) + feat_num_oov(
            existing_features,
            statistics['target_vocab'],
            lambda x: x['text2'],
            unwrap(sys_info.target_tokenizer),
        )

    # training set dependent features (this could be merged into the above one for
    # further optimization)
    def _get_fre_rank(
        self, sys_info: SysOutputInfo, existing_features: dict, statistics: Any
    ):
        return feat_freq_rank(
            existing_features,
            statistics['source_vocab'],
            lambda x: x['text1'],
            unwrap(sys_info.source_tokenizer),
        ) + feat_freq_rank(
            existing_features,
            statistics['target_vocab'],
            lambda x: x['text2'],
            unwrap(sys_info.target_tokenizer),
        )
