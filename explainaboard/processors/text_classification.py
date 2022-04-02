from __future__ import annotations

from typing import Any

from datalabs import aggregating
from tqdm import tqdm

from explainaboard import feature
from explainaboard.processors.processor import Processor
from explainaboard.processors.processor_registry import register_processor
from explainaboard.tasks import TaskType
import explainaboard.utils.feature_funcs
from explainaboard.utils.feature_funcs import get_basic_words, get_lexical_richness
from explainaboard.utils.spacy_loader import get_named_entities


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
                "true_label": feature.ClassLabel(names=["1", "0"], is_bucket=False),
                "predicted_label": feature.ClassLabel(
                    names=["1", "0"], is_bucket=False
                ),
                "label": feature.Value(
                    dtype="string",
                    description="category",
                    is_bucket=True,
                    bucket_info=feature.BucketInfo(
                        method="bucket_attribute_discrete_value", number=4, setting=1
                    ),
                ),
                "sentence_length": feature.Value(
                    dtype="float",
                    description="text length",
                    is_bucket=True,
                    bucket_info=feature.BucketInfo(
                        method="bucket_attribute_specified_bucket_value",
                        number=4,
                        setting=(),
                    ),
                ),
                "token_number": feature.Value(
                    dtype="float",
                    description="the number of chars",
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
                "entity_number": feature.Value(
                    dtype="float",
                    description="the number of entities",
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
    def default_metrics(cls) -> list[str]:
        return ["Accuracy"]

    # --- Feature functions accessible by ExplainaboardBuilder._get_feature_func()
    def _get_sentence_length(self, existing_features: dict):
        return len(self._tokenizer(existing_features["text"]))

    def _get_token_number(self, existing_feature: dict):
        return len(existing_feature["text"])

    def _get_entity_number(self, existing_feature: dict):
        return len(get_named_entities(existing_feature["text"]))

    def _get_label(self, existing_feature: dict):
        return existing_feature["true_label"]

    def _get_basic_words(self, existing_feature: dict):
        return get_basic_words(existing_feature["text"])

    def _get_lexical_richness(self, existing_feature: dict):
        return get_lexical_richness(existing_feature["text"])

    # training set dependent features
    def _get_num_oov(self, existing_features: dict, statistics: Any):
        return explainaboard.utils.feature_funcs.feat_num_oov(
            existing_features, statistics, lambda x: x['text'], self._tokenizer
        )

    # training set dependent features
    # (this could be merged into the above one for further optimization)
    def _get_fre_rank(self, existing_features: dict, statistics: Any):
        return explainaboard.utils.feature_funcs.feat_freq_rank(
            existing_features, statistics, lambda x: x['text'], self._tokenizer
        )

    # training set dependent features
    def _get_length_fre(self, existing_features: dict, statistics: Any):
        length_fre = 0
        length = len(self._tokenizer(existing_features["text"]))

        if length in statistics['length_fre'].keys():
            length_fre = statistics['length_fre'][length]

        return length_fre

    # --- End feature functions

    @aggregating()
    def _statistics_func(self, samples):
        """
        Input:
        samples: [{
         "text":
         "label":
        }]
        """

        # TODO(gneubig):
        # BEWARE THIS IS HACKY. This should use the same tokenizer as the processor.
        # tokenizer = SingleSpaceTokenizer()

        vocab = {}
        length_fre = {}
        total_samps = 0
        for sample in tqdm(samples):
            text = sample["text"]
            tokens = self._tokenizer(text)
            length = len(tokens)

            if length in length_fre.keys():
                length_fre[length] += 1
            else:
                length_fre[length] = 1

            # update vocabulary
            for w in tokens:
                if w in vocab.keys():
                    vocab[w] += 1
                else:
                    vocab[w] = 1

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
