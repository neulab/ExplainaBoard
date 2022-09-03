from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from explainaboard import TaskType
from explainaboard.analysis import feature
from explainaboard.analysis.analyses import (
    Analysis,
    AnalysisLevel,
    BucketAnalysis,
    ComboCountAnalysis,
)
from explainaboard.analysis.feature import FeatureType
from explainaboard.analysis.feature_funcs import (
    accumulate_vocab_from_samples,
    count_tokens,
    feat_freq_rank,
    feat_num_oov,
    get_similarity_by_sacrebleu,
)
from explainaboard.info import SysOutputInfo
from explainaboard.metrics.accuracy import AccuracyConfig
from explainaboard.metrics.metric import MetricConfig
from explainaboard.processors.processor import Processor
from explainaboard.processors.processor_registry import register_processor
from explainaboard.utils.typing_utils import unwrap


@register_processor(TaskType.text_pair_classification)
class TextPairClassificationProcessor(Processor):
    @classmethod
    def task_type(cls) -> TaskType:
        return TaskType.text_pair_classification

    def default_analysis_levels(self) -> list[AnalysisLevel]:
        features: dict[str, FeatureType] = {
            "text1": feature.Value(
                dtype="string",
                description="the first text",
            ),
            "text2": feature.Value(
                dtype="string",
                description="the second text",
            ),
            "true_label": feature.Value(
                dtype="string",
                description="the true label of the input",
            ),
            "predicted_label": feature.Value(
                dtype="string",
                description="the predicted label",
            ),
            "text1_length": feature.Value(
                dtype="float",
                description="text1 length in tokens",
                func=lambda info, x, c: count_tokens(info, x['text1'], side='source'),
            ),
            "text2_length": feature.Value(
                dtype="float",
                description="text2 length in tokens",
                func=lambda info, x, c: count_tokens(info, x['text2'], side='target'),
            ),
            "similarity": feature.Value(
                dtype="float",
                description="the two texts' similarity",
                func=lambda info, x, c: get_similarity_by_sacrebleu(
                    x['text1'], x['text2']
                ),
            ),
            "text1_divided_text2": feature.Value(
                dtype="float",
                description="ratio of two texts' lengths",
                func=lambda info, x, c: c.features['text1_length']
                / c.features['text2_length'],
            ),
            "num_oov": feature.Value(
                dtype="float",
                description="the number of out-of-vocabulary words",
                require_training_set=True,
                func=lambda info, x, c, stat: feat_num_oov(
                    info, x['text1'], stat['source_vocab'], side='source'
                )
                + feat_num_oov(info, x['text2'], stat['target_vocab'], side='target'),
            ),
            "fre_rank": feature.Value(
                dtype="float",
                description=(
                    "the average rank of each word based on its frequency in "
                    "training set"
                ),
                require_training_set=True,
                func=lambda info, x, c, stat: feat_freq_rank(
                    info, x['text1'], stat['source_vocab_rank'], side='source'
                )
                + feat_freq_rank(
                    info, x['text2'], stat['target_vocab_rank'], side='target'
                ),
            ),
        }

        return [
            AnalysisLevel(
                name='example',
                features=features,
                metric_configs=self.default_metrics(),
            )
        ]

    def default_analyses(self) -> list[Analysis]:
        features = self.default_analysis_levels()[0].features
        # Create analyses
        analyses: list[Analysis] = [
            BucketAnalysis(
                level="example",
                description=features['true_label'].description,
                feature="true_label",
                method="discrete",
                number=15,
            ),
            ComboCountAnalysis(
                level="example",
                description="confusion matrix",
                features=("true_label", "predicted_label"),
            ),
        ]
        analyses.extend(self.continuous_feature_analyses())
        return analyses

    @classmethod
    def default_metrics(
        cls, level='example', source_language=None, target_language=None
    ) -> list[MetricConfig]:
        return [AccuracyConfig(name='Accuracy')]

    def _statistics_func(self, samples: Iterable[Any], sys_info: SysOutputInfo):

        samples_list = list(samples)
        source_vocab, source_vocab_rank = accumulate_vocab_from_samples(
            samples_list,
            lambda x: x['text1'],
            unwrap(sys_info.source_tokenizer),
        )

        target_vocab, target_vocab_rank = accumulate_vocab_from_samples(
            samples_list,
            lambda x: x["text2"],
            unwrap(sys_info.target_tokenizer),
        )

        return {
            'source_vocab': source_vocab,
            'source_vocab_rank': source_vocab_rank,
            'target_vocab': target_vocab,
            'target_vocab_rank': target_vocab_rank,
        }
