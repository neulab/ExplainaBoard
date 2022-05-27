from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from datalabs import aggregating

from explainaboard import feature, TaskType
from explainaboard.info import SysOutputInfo
from explainaboard.metric import AccuracyConfig, MetricConfig
from explainaboard.processors.processor import Processor
from explainaboard.processors.processor_registry import register_processor
import explainaboard.utils.feature_funcs
from explainaboard.utils.feature_funcs import get_basic_words, get_lexical_richness
from explainaboard.utils.logging import progress
from explainaboard.utils.typing_utils import unwrap


@register_processor(TaskType.text_classification)
class TextClassificationProcessor(Processor):
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
                    description="category",
                    is_bucket=True,
                    bucket_info=feature.BucketInfo(
                        method="bucket_attribute_discrete_value", number=4, setting=1
                    ),
                ),
                "text_length": feature.Value(
                    dtype="float",
                    description="text length in tokens",
                    is_bucket=True,
                    bucket_info=feature.BucketInfo(
                        method="bucket_attribute_specified_bucket_value",
                        number=4,
                        setting=(),
                    ),
                ),
                "text_chars": feature.Value(
                    dtype="float",
                    description="text length in characters",
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
                "length_fre": feature.Value(
                    dtype="float",
                    description="the frequency of text length in training set",
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

    # --- Feature functions accessible by ExplainaboardBuilder._get_feature_func()
    def _get_text_length(self, sys_info: SysOutputInfo, existing_features: dict):
        return len(unwrap(sys_info.source_tokenizer)(existing_features["text"]))

    def _get_text_chars(self, sys_info: SysOutputInfo, existing_feature: dict):
        return len(existing_feature["text"])

    def _get_label(self, sys_info: SysOutputInfo, existing_feature: dict):
        return existing_feature["true_label"]

    def _get_basic_words(self, sys_info: SysOutputInfo, existing_feature: dict):
        return get_basic_words(existing_feature["text"])

    def _get_lexical_richness(self, sys_info: SysOutputInfo, existing_feature: dict):
        return get_lexical_richness(existing_feature["text"])

    # training set dependent features
    def _get_num_oov(
        self, sys_info: SysOutputInfo, existing_features: dict, statistics: Any
    ):
        return explainaboard.utils.feature_funcs.feat_num_oov(
            existing_features,
            statistics,
            lambda x: x['text'],
            unwrap(sys_info.source_tokenizer),
        )

    # training set dependent features
    # (this could be merged into the above one for further optimization)
    def _get_fre_rank(
        self, sys_info: SysOutputInfo, existing_features: dict, statistics: Any
    ):
        return explainaboard.utils.feature_funcs.feat_freq_rank(
            existing_features,
            statistics,
            lambda x: x['text'],
            unwrap(sys_info.source_tokenizer),
        )

    # training set dependent features
    def _get_length_fre(
        self, sys_info: SysOutputInfo, existing_features: dict, statistics: Any
    ):
        length = len(unwrap(sys_info.source_tokenizer)(existing_features["text"]))
        return statistics['length_fre'].get(str(length), 0)

    # --- End feature functions

    @aggregating()
    def _statistics_func(self, samples: Iterator, sys_info: SysOutputInfo):
        vocab: dict[str, float] = {}
        length_fre: dict[str, float] = {}
        total_samps = 0
        tokenizer = unwrap(sys_info.source_tokenizer)
        for sample in progress(samples):
            text = sample["text"]
            tokens = tokenizer(text)
            length = str(len(tokens))

            length_fre[length] = length_fre.get(length, 0.0) + 1.0

            # update vocabulary
            for w in tokens:
                vocab[w] = vocab.get(w, 0.0) + 1.0

            total_samps += 1

        # the rank of each word based on its frequency
        sorted_dict = {
            key: rank
            for rank, key in enumerate(sorted(set(vocab.values()), reverse=True), 1)
        }
        vocab_rank = {k: sorted_dict[v] for k, v in vocab.items()}

        for k, v in length_fre.items():
            length_fre[k] = v * 1.0 / total_samps

        return {"vocab": vocab, "vocab_rank": vocab_rank, "length_fre": length_fre}
