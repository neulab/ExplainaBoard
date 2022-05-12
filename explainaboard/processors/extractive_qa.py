from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from datalabs import aggregating

from explainaboard import feature, TaskType
from explainaboard.info import SysOutputInfo
from explainaboard.metric import ExactMatchQAConfig, F1ScoreQAConfig, MetricConfig
from explainaboard.processors.processor import Processor
from explainaboard.processors.processor_registry import register_processor
import explainaboard.utils.feature_funcs
from explainaboard.utils.typing_utils import unwrap


@register_processor(TaskType.qa_extractive)
class QAExtractiveProcessor(Processor):
    @classmethod
    def task_type(cls) -> TaskType:
        return TaskType.qa_extractive

    @classmethod
    def default_features(cls) -> feature.Features:
        return feature.Features(
            {
                "title": feature.Value("string"),
                "context": feature.Value("string"),
                "question": feature.Value("string"),
                "id": feature.Value("string"),
                "answers": feature.Sequence(feature=feature.Value("string")),
                "predicted_answers": feature.Value("string"),
                "context_length": feature.Value(
                    dtype="float",
                    description="context length",
                    is_bucket=True,
                    bucket_info=feature.BucketInfo(
                        method="bucket_attribute_specified_bucket_value",
                        number=4,
                        setting=(),
                    ),
                ),
                "question_length": feature.Value(
                    dtype="float",
                    description="question length",
                    is_bucket=True,
                    bucket_info=feature.BucketInfo(
                        method="bucket_attribute_specified_bucket_value",
                        number=4,
                        setting=(),
                    ),
                ),
                "answer_length": feature.Value(
                    dtype="float",
                    description="answer length",
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
        if source_language != target_language:
            raise ValueError(
                'Source and target language must be equal for extractive '
                f'QA, but got {source_language} and {target_language}'
            )
        return [
            F1ScoreQAConfig(
                name='F1',
                source_language=source_language,
                target_language=target_language,
            ),
            ExactMatchQAConfig(
                name='ExactMatch',
                source_language=source_language,
                target_language=target_language,
            ),
        ]

    @aggregating()
    def _statistics_func(self, samples: Iterator, sys_info: SysOutputInfo):
        return explainaboard.utils.feature_funcs.accumulate_vocab_from_samples(
            samples, lambda x: x['context'], unwrap(sys_info.source_tokenizer)
        )

    # --- Feature functions accessible by ExplainaboardBuilder._get_feature_func()
    def _get_context_length(self, sys_info: SysOutputInfo, existing_features: dict):
        return len(unwrap(sys_info.source_tokenizer)(existing_features["context"]))

    def _get_question_length(self, sys_info: SysOutputInfo, existing_features: dict):
        return len(unwrap(sys_info.source_tokenizer)(existing_features["question"]))

    def _get_answer_length(self, sys_info: SysOutputInfo, existing_features: dict):
        if isinstance(existing_features["answers"]["text"], list):
            return len(
                unwrap(sys_info.source_tokenizer)(
                    existing_features["answers"]["text"][0]
                )
            )
        else:
            return len(
                unwrap(sys_info.source_tokenizer)(existing_features["answers"]["text"])
            )

    def _get_sim_context_question(
        self, sys_info: SysOutputInfo, existing_features: dict
    ):

        references = existing_features["context"]
        hypothesis = existing_features["question"]

        res_json = self._get_eaas_client().bleu([[references]], [hypothesis], lang="en")
        return res_json["corpus_bleu"]

    # training set dependent features (could be merged for optimization?)
    def _get_num_oov(
        self, sys_info: SysOutputInfo, existing_features: dict, statistics: Any
    ):
        return explainaboard.utils.feature_funcs.feat_num_oov(
            existing_features,
            statistics,
            lambda x: x['context'],
            unwrap(sys_info.source_tokenizer),
        )

    def _get_fre_rank(
        self, sys_info: SysOutputInfo, existing_features: dict, statistics: Any
    ):
        return explainaboard.utils.feature_funcs.feat_freq_rank(
            existing_features,
            statistics,
            lambda x: x['context'],
            unwrap(sys_info.source_tokenizer),
        )

    # --- End feature functions

    def _get_true_label(self, data_point: dict):
        return data_point["answers"]["text"]

    def _get_predicted_label(self, data_point: dict):
        return data_point["predicted_answers"]["text"]
