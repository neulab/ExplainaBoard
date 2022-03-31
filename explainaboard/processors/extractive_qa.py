from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from datalabs import aggregating

from explainaboard import feature
from explainaboard.processors.processor import Processor
from explainaboard.processors.processor_registry import register_processor
from explainaboard.tasks import TaskType
import explainaboard.utils.feature_funcs


@register_processor(TaskType.question_answering_extractive)
class QAExtractiveProcessor(Processor):
    @classmethod
    def task_type(cls) -> TaskType:
        return TaskType.question_answering_extractive

    @classmethod
    def default_features(cls) -> feature.Features:
        return feature.Features(
            {
                "title": feature.Value("string"),
                "context": feature.Value("string"),
                "question": feature.Value("string"),
                "id": feature.Value("string"),
                "answers": feature.Sequence(feature.Value("string")),
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
        return ["F1ScoreQA", "ExactMatchQA"]

    @aggregating()
    def _statistics_func(self, samples: Iterator):
        """
        Input:
        samples: [{
         "id":str
         "context":str
         "question":str
         "answers":Dict
         "options"
        }]
        """

        # TODO(gneubig):
        # BEWARE THIS IS HACKY. This should use the same tokenizer as the processor.
        # tokenizer = SingleSpaceTokenizer()

        return explainaboard.utils.feature_funcs.accumulate_vocab_from_samples(
            samples, lambda x: x['context'], self._tokenizer
        )

    # --- Feature functions accessible by ExplainaboardBuilder._get_feature_func()
    def _get_context_length(self, existing_features: dict):
        return len(self._tokenizer(existing_features["context"]))

    def _get_question_length(self, existing_features: dict):
        return len(self._tokenizer(existing_features["question"]))

    def _get_answer_length(self, existing_features: dict):
        if isinstance(existing_features["answers"]["text"], list):
            return len(self._tokenizer(existing_features["answers"]["text"][0]))
        else:
            return len(self._tokenizer(existing_features["answers"]["text"]))

    def _get_sim_context_question(self, existing_features: dict):

        references = existing_features["context"]
        hypothesis = existing_features["question"]

        res_json = self._get_eaas_client().bleu([[references]], [hypothesis], lang="en")
        return res_json["corpus_bleu"]

    # training set dependent features (could be merged for optimization?)
    def _get_num_oov(self, existing_features: dict, statistics: Any):
        return explainaboard.utils.feature_funcs.feat_num_oov(
            existing_features, statistics, lambda x: x['context'], self._tokenizer
        )

    def _get_fre_rank(self, existing_features: dict, statistics: Any):
        return explainaboard.utils.feature_funcs.feat_freq_rank(
            existing_features, statistics, lambda x: x['context'], self._tokenizer
        )

    # --- End feature functions

    def _get_true_label(self, data_point: dict):
        return data_point["answers"]["text"]

    def _get_predicted_label(self, data_point: dict):
        return data_point["predicted_answers"]["text"]
