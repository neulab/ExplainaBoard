from __future__ import annotations

from typing import Any

from datalabs import aggregating

from explainaboard import feature
from explainaboard.processors.processor import Processor
from explainaboard.processors.processor_registry import register_processor
from explainaboard.tasks import TaskType
import explainaboard.utils.feature_funcs


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
                "true_label": feature.ClassLabel(names=["1", "0"], is_bucket=False),
                "predicted_label": feature.ClassLabel(
                    names=["1", "0"], is_bucket=False
                ),
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
                        "the average rank of each work based on its frequency in "
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
    def default_metrics(cls) -> list[str]:
        return ["Accuracy"]

    @aggregating()
    def _statistics_func(self, samples):
        # TODO(gneubig):
        # BEWARE THIS IS HACKY. This should use the same tokenizer as the processor.
        tokenizer = self._tokenizer

        return explainaboard.utils.feature_funcs.accumulate_vocab_from_samples(
            samples, lambda x: x['text1'] + x['text2'], tokenizer
        )

    # --- Feature functions accessible by ExplainaboardBuilder._get_feature_func()
    def _get_similarity(self, existing_features: dict):
        return explainaboard.utils.feature_funcs.get_similarity_by_sacrebleu(
            existing_features["text1"], existing_features["text2"]
        )

    def _get_text1_length(self, existing_features: dict):
        return len(self._tokenizer(existing_features["text1"]))

    def _get_text2_length(self, existing_feature: dict):
        return len(self._tokenizer(existing_feature["text2"]))

    def _get_text1_divided_text2(self, existing_feature: dict):
        return (
            len(self._tokenizer(existing_feature["text1"]))
            * 1.0
            / len(self._tokenizer(existing_feature["text2"]))
        )

    def _get_label(self, existing_feature: dict):
        # print(f"print_existing_feature: \t {existing_feature}")
        return existing_feature["true_label"]

    # training set dependent features
    def _get_num_oov(self, existing_features: dict, statistics: Any):
        return explainaboard.utils.feature_funcs.feat_num_oov(
            existing_features,
            statistics,
            lambda x: x['text1'] + x['text2'],
            self._tokenizer,
        )

    # training set dependent features (this could be merged into the above one for
    # further optimization)
    def _get_fre_rank(self, existing_features: dict, statistics: Any):
        return explainaboard.utils.feature_funcs.feat_freq_rank(
            existing_features,
            statistics,
            lambda x: x['text1'] + x['text2'],
            self._tokenizer,
        )
