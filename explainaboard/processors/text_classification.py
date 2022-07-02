from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Sequence

from datalabs import aggregating

from explainaboard import TaskType
from explainaboard.analysis import feature
from explainaboard.info import SysOutputInfo
from explainaboard.analysis.analyses import BucketAnalysis, AnalysisLevel, Analysis
from explainaboard.metrics.accuracy import AccuracyConfig
from explainaboard.metrics.metric import MetricConfig
from explainaboard.processors.processor import Processor
from explainaboard.processors.processor_registry import register_processor
from explainaboard.analysis.feature_funcs import get_basic_words, get_lexical_richness, count_tokens, feat_num_oov, feat_freq_rank, feat_length_freq
from explainaboard.utils.logging import progress
from explainaboard.utils.typing_utils import unwrap


@register_processor(TaskType.text_classification)
class TextClassificationProcessor(Processor):
    @classmethod
    def task_type(cls) -> TaskType:
        return TaskType.text_classification

    @classmethod
    def default_analyses(cls) -> list[AnalysisLevel]:
        features = {
            "text": feature.Value(
                dtype="string",
                description="the text of the example",
            ),
            "true_label": feature.Value(
                dtype="string",
                description="the true label of the input",
            ),
            "predicted_label": feature.Value(
                dtype="string",
                description="the predicted label",
            ),
            "text_length": feature.Value(
                dtype="float",
                description="text length in tokens",
                func=lambda info, x: count_tokens(info, x['text']),
            ),
            "text_chars": feature.Value(
                dtype="float",
                description="text length in characters",
                func=lambda info, x: len(x['text']),
            ),
            "basic_words": feature.Value(
                dtype="float",
                description="the ratio of basic words",
                func=lambda info, x: get_basic_words(x['text']),
            ),
            "lexical_richness": feature.Value(
                dtype="float",
                description="lexical diversity",
                func=lambda info, x: get_lexical_richness(x['text']),
            ),
            "num_oov": feature.Value(
                dtype="float",
                description="the number of out-of-vocabulary words",
                require_training_set=True,
                func=lambda info, x, stat: feat_num_oov(info, x['text'], stat)
            ),
            "fre_rank": feature.Value(
                dtype="float",
                description=(
                    "the average rank of each word based on its frequency in "
                    "training set"
                ),
                require_training_set=True,
                func=lambda info, x, stat: feat_freq_rank(info, x['text'], stat)
            ),
            "length_fre": feature.Value(
                dtype="float",
                description="the frequency of text length in training set",
                require_training_set=True,
                func=lambda info, x, stat: feat_length_freq(info, x['text'], stat)
            )
        }
        continuous_features = [k for k, v in features.items() if ('float' in unwrap(v.dtype))]
        print(f'continuous_features = {continuous_features}')
        analyses: Sequence[Analysis] = ([
                BucketAnalysis(
                    feature="true_label", method="discrete", number=15,
                )] +
                [BucketAnalysis(x, method="continuous") for x in continuous_features])

        return [AnalysisLevel(
            name='example',
            features=features,
            metric_configs=cls.default_metrics(),
            analyses=analyses
        )]

    @classmethod
    def default_metrics(
        cls, source_language=None, target_language=None
    ) -> list[MetricConfig]:
        return [AccuracyConfig(name='Accuracy')]


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
