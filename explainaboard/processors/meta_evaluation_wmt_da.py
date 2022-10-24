"""A processor for the WMT Metrics meta-evaluation task."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from explainaboard import TaskType
from explainaboard.analysis import feature
from explainaboard.analysis.analyses import Analysis, AnalysisLevel
from explainaboard.analysis.feature import FeatureType
from explainaboard.analysis.feature_funcs import count_tokens
from explainaboard.info import SysOutputInfo
from explainaboard.metrics.meta_evaluation import (
    KtauCorrelationWMTDAConfig,
    PearsonCorrelationWMTDAConfig,
)
from explainaboard.metrics.metric import MetricConfig
from explainaboard.processors.processor import Processor
from explainaboard.utils.language_utils import (
    is_chinese_lang_code,
    is_japanese_lang_code,
)
from explainaboard.utils.tokenizer import SacreBleuTokenizer, Tokenizer


class MetaEvaluationWMTDAProcessor(Processor):
    """A processor for the WMT meta-evaluation task."""

    @classmethod
    def task_type(cls) -> TaskType:
        """See Processor.task_type."""
        return TaskType.meta_evaluation_wmt_da

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
            "sys_name": feature.Value(
                dtype=feature.DataType.STRING,
                description="the name of the system",
            ),
            "seg_id": feature.Value(
                dtype=feature.DataType.STRING,
                description="the ID of the segment",
            ),
            "test_set": feature.Value(
                dtype=feature.DataType.STRING,
                description="the set from which the example came from",
            ),
            "src": feature.Value(
                dtype=feature.DataType.STRING,
                description="the source sentence",
            ),
            "ref": feature.Value(
                dtype=feature.DataType.STRING,
                description="the reference sentence",
            ),
            "sys": feature.Value(
                dtype=feature.DataType.STRING,
                description="the system output",
            ),
            "manual_raw": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="the raw score provided by annotators",
            ),
            "manual_z": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="the z-normalized score provided by annotators",
            ),
            "auto_score": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="the score provided by the automatic system",
            ),
            "src_length": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="source length",
                func=lambda info, x, c: count_tokens(info, x["src"]),
            ),
            "ref_length": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="reference length",
                func=lambda info, x, c: count_tokens(info, x["ref"], side="target"),
            ),
            "sys_length": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="system output length",
                func=lambda info, x, c: count_tokens(info, x["ref"], side="target"),
            ),
            "src_divided_ref": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="ratio of source length to reference length",
                func=lambda info, x, c: c.features["src_length"]
                / c.features["ref_length"],
            ),
            "sys_divided_ref": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="ratio of system output length to reference length",
                func=lambda info, x, c: c.features["sys_length"]
                / c.features["ref_length"],
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
            "SegKtauCorr": KtauCorrelationWMTDAConfig(
                group_by="segment", use_z_score=False
            ),
            "SysPearsonCorr": PearsonCorrelationWMTDAConfig(group_by="system"),
        }

    # --- Feature functions accessible by ExplainaboardBuilder._get_feature_func()
    def _get_true_label(self, data_point: dict):
        """See processor._get_true_label."""
        return [
            data_point["sys_name"],
            data_point["seg_id"],
            data_point["manual_raw"],
            data_point["manual_z"],
        ]

    def _get_predicted_label(self, data_point: dict):
        """See processor._get_predicted_label."""
        return data_point["auto_score"]
