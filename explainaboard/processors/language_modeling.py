"""A processor for the language modeling task."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from explainaboard import TaskType
from explainaboard.analysis import feature
from explainaboard.analysis.analyses import Analysis, AnalysisLevel, BucketAnalysis
from explainaboard.analysis.case import AnalysisCase, AnalysisCaseSpan
from explainaboard.analysis.feature import DataType, FeatureType, Value
from explainaboard.analysis.feature_funcs import (
    cap_feature,
    count_tokens,
    feat_freq_rank,
    feat_length_freq,
    feat_num_oov,
)
from explainaboard.info import SysOutputInfo
from explainaboard.metrics.log_prob import LogProbConfig
from explainaboard.metrics.metric import MetricConfig, MetricStats, SimpleMetricStats
from explainaboard.processors.processor import Processor
from explainaboard.utils.logging import progress
from explainaboard.utils.typing_utils import unwrap


class LanguageModelingProcessor(Processor):
    """A processor for the language modeling task."""

    @classmethod
    def task_type(cls) -> TaskType:
        """See Processor.task_type."""
        return TaskType.language_modeling

    def default_analysis_levels(self) -> list[AnalysisLevel]:
        """See Processor.default_analysis_levels."""
        examp_features: dict[str, FeatureType] = {
            "text": feature.Value(dtype=feature.DataType.STRING),
            "log_probs": feature.Value(dtype=feature.DataType.STRING),
            "text_length": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="text length in tokens",
                func=lambda info, x, c: count_tokens(info, x["text"]),
            ),
            "text_chars": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="text length in characters",
                func=lambda info, x, c: len(x["text"]),
            ),
            "num_oov": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="the number of out-of-vocabulary words",
                require_training_set=True,
                func=lambda info, x, c, stat: feat_num_oov(
                    info, x["text"], stat["vocab"]
                ),
            ),
            "fre_rank": feature.Value(
                dtype=feature.DataType.FLOAT,
                description=(
                    "the average rank of each word based on its frequency in "
                    "training set"
                ),
                require_training_set=True,
                func=lambda info, x, c, stat: feat_freq_rank(
                    info, x["text"], stat["vocab_rank"]
                ),
            ),
            "length_fre": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="the frequency of text length in training set",
                require_training_set=True,
                func=lambda info, x, c, stat: feat_length_freq(
                    info, x["text"], stat["length_fre"]
                ),
            ),
        }

        tok_features: dict[str, FeatureType] = {
            "tok_log_prob": feature.Value(
                dtype=feature.DataType.FLOAT,
                description=("log probability of the token according to the LM"),
            ),
            "tok_capitalness": feature.Value(
                dtype=feature.DataType.STRING,
                description=(
                    "The capitalness of an token. For example, "
                    "first_caps represents only the first character of "
                    "the token is capital. full_caps denotes all "
                    "characters of the token are capital"
                ),
                func=lambda info, x, c: cap_feature(c.text),
            ),
            "tok_position": feature.Value(
                dtype=feature.DataType.FLOAT,
                description=("The relative position of a token in a sentence"),
                func=lambda info, x, c: c.token_span[0] / count_tokens(info, x["text"]),
            ),
            "tok_chars": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="The number of characters in a token",
                func=lambda info, x, c: len(c.text),
            ),
            # TODO(gneubig): commented out because probably less important
            # "tok_test_freq": feature.Value(
            #     dtype=feature.DataType.FLOAT,
            #     description="tok frequency in the test set",
            #     require_training_set=False,
            #     func=...
            # ),
            "tok_train_freq": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="tok frequency in the training set",
                require_training_set=True,
                func=lambda info, x, c, stat: stat["vocab"].get(c.text, 0.0),
            ),
        }

        return [
            AnalysisLevel(
                name="example",
                features=examp_features,
                metric_configs=self.default_metrics(level="example"),
            ),
            AnalysisLevel(
                name="token",
                features=tok_features,
                metric_configs=self.default_metrics(level="token"),
            ),
        ]

    def default_analyses(self) -> list[Analysis]:
        """See Processor.default_analyses."""
        analyses: list[Analysis] = []
        analysis_levels = self.default_analysis_levels()
        for lev in analysis_levels:
            for k, v in lev.features.items():
                if (
                    isinstance(v, Value)
                    and v.dtype == DataType.FLOAT
                    and k != "tok_log_prob"
                ):
                    analyses.append(
                        BucketAnalysis(
                            level=lev.name,
                            description=lev.features[k].description,
                            feature=k,
                            method="continuous",
                        )
                    )
        return analyses

    def _gen_cases_and_stats(
        self,
        sys_info: SysOutputInfo,
        sys_output: list[dict],
        statistics: Any,
        analysis_level: AnalysisLevel,
    ) -> tuple[list[AnalysisCase], dict[str, MetricStats]]:
        if analysis_level.name == "example":
            return super()._gen_cases_and_stats(
                sys_info, sys_output, statistics, analysis_level
            )
        elif analysis_level.name != "token":
            raise ValueError(f"{analysis_level.name}-level analysis not supported")
        # Do tok-level analysis
        cases: list[AnalysisCase] = []
        # Calculate features
        for i, output in progress(
            enumerate(sys_output), desc="calculating tok-level features"
        ):
            # get the tokens and scores from each sentence
            toks = output["text"].split(" ")
            probs = [float(x) for x in output["log_probs"].split(" ")]
            # analysis cases
            curr_char = 0
            for j, (tok, prob) in enumerate(zip(toks, probs)):
                next_char = curr_char + len(tok)
                case = AnalysisCaseSpan(
                    sample_id=i,
                    features={"tok_log_prob": prob},
                    token_span=(j, j + 1),
                    char_span=(curr_char, next_char),
                    text=tok,
                    orig_str="source",
                )
                curr_char = next_char + 1
                for feat_name, feat_spec in analysis_level.features.items():
                    if feat_spec.func is None:
                        pass
                    elif not feat_spec.require_training_set:
                        case.features[feat_name] = feat_spec.func(
                            sys_info, output, case
                        )
                    elif statistics is not None:
                        case.features[feat_name] = feat_spec.func(
                            sys_info, output, case, statistics
                        )
                cases.append(case)
        metric_stats: dict[str, MetricStats] = {
            "Perplexity": SimpleMetricStats(
                np.array([x.features["tok_log_prob"] for x in cases])
            ),
            "LogProb": SimpleMetricStats(
                np.array([x.features["tok_log_prob"] for x in cases])
            ),
        }
        return cases, metric_stats

    @classmethod
    def default_metrics(
        cls,
        level: str = "example",
        source_language: str | None = None,
        target_language: str | None = None,
    ) -> dict[str, MetricConfig]:
        """See Processor.default_metrics."""
        return {
            "Perplexity": LogProbConfig(ppl=True),
            "LogProb": LogProbConfig(ppl=False),
        }

    def _get_true_label(self, data_point: dict):
        """See processor._get_true_label."""
        return None

    def _get_predicted_label(self, data_point: dict):
        """See processor._get_predicted_label."""
        return [float(x) for x in data_point["log_probs"].split(" ")]

    def _statistics_func(self, samples: Iterable[Any], sys_info: SysOutputInfo):
        vocab: dict[str, float] = {}
        length_fre: dict[int, float] = {}
        total_samps = 0
        tokenizer = unwrap(sys_info.source_tokenizer)
        for sample in progress(samples):
            text = sample["text"]
            tokens = tokenizer(text)
            length = len(tokens)

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
