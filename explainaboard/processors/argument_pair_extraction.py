"""A processor for the argument pair extraction task."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast

from explainaboard import TaskType
from explainaboard.analysis import feature
from explainaboard.analysis.analyses import Analysis, AnalysisLevel
from explainaboard.analysis.case import AnalysisCase, AnalysisCaseLabeledArgumentPair
from explainaboard.analysis.feature import FeatureType
from explainaboard.analysis.feature_funcs import accumulate_vocab_from_samples
from explainaboard.info import SysOutputInfo
from explainaboard.metrics.f1_score import APEF1ScoreConfig, F1ScoreConfig
from explainaboard.metrics.metric import MetricConfig, MetricStats
from explainaboard.processors.processor import Processor
from explainaboard.utils.logging import progress
from explainaboard.utils.span_utils import ArgumentPair, ArgumentPairOps
from explainaboard.utils.typing_utils import unwrap


class ArgumentPairExtractionProcessor(Processor):
    """A processor for the argument pair extraction task."""

    @classmethod
    def task_type(cls) -> TaskType:
        """See Processor.task_type."""
        return TaskType.argument_pair_extraction

    def __init__(self):
        """Constructor."""
        super().__init__()
        self._argument_pair_ops: ArgumentPairOps = ArgumentPairOps()

    _DEFAULT_TAG = "O"

    @classmethod
    def default_metrics(
        cls,
        level: str = "example",
        source_language: str | None = None,
        target_language: str | None = None,
    ) -> dict[str, MetricConfig]:
        """See Processor.default_metrics."""
        defaults: dict[str, dict[str, MetricConfig]] = {
            "example": {
                "F1": APEF1ScoreConfig(
                    source_language=source_language,
                    target_language=target_language,
                )
            },
            "block": {
                "F1": F1ScoreConfig(
                    source_language=source_language,
                    target_language=target_language,
                    ignore_classes=[cls._DEFAULT_TAG],
                )
            },
        }
        return defaults[level]

    def default_analysis_levels(self) -> list[AnalysisLevel]:
        """See Processor.default_analysis_levels."""
        features = {
            "sentences": feature.Sequence(
                feature=feature.Value(dtype=feature.DataType.STRING)
            ),
            "true_tags": feature.Sequence(
                feature=feature.Value(dtype=feature.DataType.STRING)
            ),
            "pred_tags": feature.Sequence(
                feature=feature.Value(dtype=feature.DataType.STRING)
            ),
            "num_sent": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="the number of sentences",
                func=lambda info, x, c: len(x["sentences"]),
            ),
            "text_length": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="the length of all sentences",
                func=lambda info, x, c: len(" ".join(x["sentences"])),
            ),
        }

        block_features: dict[str, FeatureType] = {
            "text": feature.Value(
                dtype=feature.DataType.STRING,
                description="text of the block",
                func=lambda info, x, c: c.text,
            ),
            "n_review_sentences": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="the number of review sentence",
                func=lambda info, x, c: c.block_review_sentences,
            ),
            "n_review_tokens": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="the number of review tokens",
                func=lambda info, x, c: c.block_review_tokens,
            ),
            "n_review_position": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="the relative position of review sentence",
                func=lambda info, x, c: c.block_review_position,
            ),
            "n_reply_sentences": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="the number of reply sentence",
                func=lambda info, x, c: c.block_reply_sentences,
            ),
            "n_reply_tokens": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="the number of reply tokens",
                func=lambda info, x, c: c.block_reply_tokens,
            ),
            "n_reply_position": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="the relative position of reply sentence",
                func=lambda info, x, c: c.block_reply_position,
            ),
        }

        return [
            AnalysisLevel(
                name="example",
                features=features,
                metric_configs=self.default_metrics(),
            ),
            AnalysisLevel(
                name="block",
                features=block_features,
                metric_configs=self.default_metrics(level="block"),
            ),
        ]

    def default_analyses(self) -> list[Analysis]:
        """See Processor.default_analyses."""
        analyses: list[Analysis] = []
        analyses.extend(self.continuous_feature_analyses())
        return analyses

    def _get_true_label(self, data_point):
        """See Processor._get_true_label."""
        return data_point["true_tags"]

    def _get_predicted_label(self, data_point):
        """See Processor._get_predicted_label."""
        return data_point["pred_tags"]

    def _statistics_func(self, samples: Iterable[Any], sys_info: SysOutputInfo):
        vocab, vocab_rank = accumulate_vocab_from_samples(
            samples,
            lambda x: " ".join(x["sentences"]),
            unwrap(sys_info.source_tokenizer),
        )

        return {"vocab": vocab, "vocab_rank": vocab_rank}

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
        elif analysis_level.name != "block":
            raise ValueError(f"{analysis_level.name}-level analysis not supported")
        # Do block-level analysis. `AnalysisCaseLabeledBlock` typing is necessary
        # otherwise an error will happen later when using `x.true_label`
        cases: list[AnalysisCaseLabeledArgumentPair] = []
        # Calculate features
        for i, output in progress(
            enumerate(sys_output), desc="calculating span-level features"
        ):
            # get the spans from each sentence
            sentences = output["sentences"]
            true_spans, pred_spans = self._argument_pair_ops.get_argument_pairs(
                output["true_tags"], output["pred_tags"], sentences
            )
            true_spans = cast(list[ArgumentPair], true_spans)
            pred_spans = cast(list[ArgumentPair], pred_spans)
            # merge the spans together
            merged_spans: dict[tuple[int, int, int, int], ArgumentPair] = {}
            for span in true_spans:
                span.block_tag = f"{span.block_tag} {self._DEFAULT_TAG}"
                merged_spans[unwrap(span.block_pos)] = span
            for span in pred_spans:
                merged_span = merged_spans.get(unwrap(span.block_pos))
                if not merged_span:
                    span.block_tag = f"{self._DEFAULT_TAG} {span.block_tag}"
                    merged_spans[unwrap(span.block_pos)] = span
                else:
                    true_tag, _ = unwrap(merged_span.block_tag).split(" ")
                    merged_span.block_tag = f"{true_tag} {span.block_tag}"
            # analysis cases
            for ms in merged_spans.values():
                true_tag, pred_tag = unwrap(ms.block_tag).split(" ")
                case = AnalysisCaseLabeledArgumentPair(
                    sample_id=i,
                    features={},
                    text=unwrap(ms.block_text),
                    true_label=true_tag,
                    predicted_label=pred_tag,
                    block_review_sentences=unwrap(ms.block_review_sentences),
                    block_review_tokens=unwrap(ms.block_review_tokens),
                    block_review_position=unwrap(ms.block_review_position),
                    block_reply_sentences=unwrap(ms.block_reply_sentences),
                    block_reply_tokens=unwrap(ms.block_reply_tokens),
                    block_reply_position=unwrap(ms.block_reply_position),
                    orig_str="source",
                )
                for feat_name, feat_spec in analysis_level.features.items():
                    if feat_spec.func is None:
                        raise ValueError(
                            f"could not find feature function for {feat_name}"
                        )
                    elif not feat_spec.require_training_set:
                        case.features[feat_name] = feat_spec.func(
                            sys_info, output, case
                        )
                    elif statistics is not None:
                        case.features[feat_name] = feat_spec.func(
                            sys_info, output, case, statistics
                        )
                cases.append(case)

        # calculate metric stats
        true_data = [x.true_label for x in cases]
        pred_data = [x.predicted_label for x in cases]
        metric_stats = {
            name: config.to_metric().calc_stats_from_data(true_data, pred_data)
            for name, config in analysis_level.metric_configs.items()
        }

        return cast(list[AnalysisCase], cases), metric_stats
