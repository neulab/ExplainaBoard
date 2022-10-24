"""A processor for the natural language generation meta-evaluation task."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from explainaboard import TaskType
from explainaboard.analysis import feature
from explainaboard.analysis.analyses import Analysis, AnalysisLevel
from explainaboard.analysis.feature import FeatureType
from explainaboard.analysis.feature_funcs import count_tokens
from explainaboard.info import SysOutputInfo
from explainaboard.metrics.meta_evaluation import CorrelationNLGConfig
from explainaboard.metrics.metric import MetricConfig
from explainaboard.processors.processor import Processor
from explainaboard.utils.language_utils import (
    is_chinese_lang_code,
    is_japanese_lang_code,
)
from explainaboard.utils.tokenizer import SacreBleuTokenizer, Tokenizer


class MetaEvaluationNLGProcessor(Processor):
    """A processor for the natural language generation meta-evaluation task."""

    @classmethod
    def task_type(cls) -> TaskType:
        """See Processor.task_type."""
        return TaskType.meta_evaluation_nlg

    def get_tokenizer(self, lang: str | None) -> Tokenizer:
        """Get a tokenizer based on the language."""
        if is_chinese_lang_code(lang):
            return SacreBleuTokenizer(variety="zh")
        elif is_japanese_lang_code(lang):
            return SacreBleuTokenizer(variety="ja-mecab")
        elif lang == "python":
            return SacreBleuTokenizer(variety="conala")
        else:
            return SacreBleuTokenizer(variety="intl")

    def default_analysis_levels(self) -> list[AnalysisLevel]:
        """See Processor.default_analysis_levels."""
        features: dict[str, FeatureType] = {
            "source": feature.Value(
                dtype=feature.DataType.STRING,
            ),
            "references": feature.Sequence(
                feature=feature.Value(dtype=feature.DataType.STRING)
            ),
            "hypotheses": feature.Sequence(
                feature=feature.Value(dtype=feature.DataType.STRING)
            ),
            "manual_scores": feature.Sequence(
                feature=feature.Value(dtype=feature.DataType.FLOAT)
            ),
            "auto_scores": feature.Sequence(
                feature=feature.Value(dtype=feature.DataType.FLOAT)
            ),
            "src_length": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="source length",
                func=lambda info, x, c: count_tokens(info, x["source"]),
            ),
        }

        return [
            AnalysisLevel(
                name="example",
                features=features,
                metric_configs=self.default_metrics(),
            )
        ]

    def default_analyses(self) -> list[Analysis]:
        """See Processor.default_analyses."""
        return self.continuous_feature_analyses()

    def _statistics_func(self, samples: Iterable[Any], sys_info: SysOutputInfo):
        return {}

    @classmethod
    def default_metrics(
        cls,
        level: str = "example",
        source_language: str | None = None,
        target_language: str | None = None,
    ) -> dict[str, MetricConfig]:
        """See Processor.default_metrics."""
        return {
            "SpearmanSampleLevelCorr": CorrelationNLGConfig(
                group_by="sample", correlation_type="spearmanr"
            ),
            "SpearmanSystemLevelCorr": CorrelationNLGConfig(
                group_by="system", correlation_type="spearmanr"
            ),
            "PearsonSampleLevelCorr": CorrelationNLGConfig(
                group_by="sample", correlation_type="pearsonr"
            ),
            "PearsonSystemLevelCorr": CorrelationNLGConfig(
                group_by="system", correlation_type="pearsonr"
            ),
        }

    # --- Feature functions accessible by ExplainaboardBuilder._get_feature_func()
    def _get_true_label(self, data_point: dict):
        """See processor._get_true_label."""
        return data_point["manual_scores"]

    def _get_predicted_label(self, data_point: dict):
        """See processor._get_predicted_label."""
        return data_point["auto_scores"]
