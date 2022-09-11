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
from explainaboard.metrics.metric import MetricConfig
from explainaboard.metrics.nlg_meta_evaluation import (
    KtauCorrelationConfig,
    PearsonCorrelationConfig,
)
from explainaboard.processors.processor import Processor
from explainaboard.processors.processor_registry import register_processor


@register_processor(TaskType.nlg_meta_evaluation)
class NLGMetaEvaluationProcessor(Processor):
    """A processor for the natural language generation meta-evaluation task."""

    @classmethod
    def task_type(cls) -> TaskType:
        """See Processor.task_type."""
        return TaskType.nlg_meta_evaluation

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
                func=lambda info, x, c: count_tokens(info, x['src']),
            ),
            "ref_length": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="reference length",
                func=lambda info, x, c: count_tokens(info, x['ref'], side='target'),
            ),
            "sys_length": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="system output length",
                func=lambda info, x, c: count_tokens(info, x['ref'], side='target'),
            ),
            "src_divided_ref": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="ratio of source length to reference length",
                func=lambda info, x, c: c.features['src_length']
                / c.features['ref_length'],
            ),
            "sys_divided_ref": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="ratio of system output length to reference length",
                func=lambda info, x, c: c.features['sys_length']
                / c.features['ref_length'],
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
        """See Processor.default_analyses."""
        return self.continuous_feature_analyses()

    def _statistics_func(self, samples: Iterable[Any], sys_info: SysOutputInfo):
        return {}

    @classmethod
    def default_metrics(
        cls, level='example', source_language=None, target_language=None
    ) -> list[MetricConfig]:
        """See Processor.default_metrics."""
        return [
            KtauCorrelationConfig(name='SegKtauCorr', group_by='segment'),
            PearsonCorrelationConfig(name='SysPearsonCorr', group_by='system'),
        ]

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
