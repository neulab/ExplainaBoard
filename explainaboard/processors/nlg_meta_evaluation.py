from __future__ import annotations

from explainaboard import TaskType
from explainaboard.analysis import feature
from explainaboard.analysis.analyses import Analysis, AnalysisLevel
from explainaboard.analysis.feature import FeatureType
from explainaboard.analysis.feature_funcs import count_tokens
from explainaboard.metrics.metric import MetricConfig
from explainaboard.metrics.nlg_meta_evaluation import (
    KtauCorrelationConfig,
    PearsonCorrelationConfig,
)
from explainaboard.processors.processor import Processor
from explainaboard.processors.processor_registry import register_processor


@register_processor(TaskType.nlg_meta_evaluation)
class NLGMetaEvaluationProcessor(Processor):
    @classmethod
    def task_type(cls) -> TaskType:
        return TaskType.nlg_meta_evaluation

    def default_analysis_levels(self) -> list[AnalysisLevel]:
        features: dict[str, FeatureType] = {
            "sys_name": feature.Value(
                dtype="string",
                description="the name of the system",
            ),
            "seg_id": feature.Value(
                dtype="string",
                description="the ID of the segment",
            ),
            "test_set": feature.Value(
                dtype="string",
                description="the set from which the example came from",
            ),
            "src": feature.Value(
                dtype="string",
                description="the source sentence",
            ),
            "ref": feature.Value(
                dtype="string",
                description="the reference sentence",
            ),
            "sys": feature.Value(
                dtype="string",
                description="the system output",
            ),
            "manual_raw": feature.Value(
                dtype="float",
                description="the raw score provided by annotators",
            ),
            "manual_z": feature.Value(
                dtype="float",
                description="the z-normalized score provided by annotators",
            ),
            "auto_score": feature.Value(
                dtype="float",
                description="the score provided by the automatic system",
            ),
            "src_length": feature.Value(
                dtype="float",
                description="source length",
                func=lambda info, x, c: count_tokens(info, x['src']),
            ),
            "ref_length": feature.Value(
                dtype="float",
                description="reference length",
                func=lambda info, x, c: count_tokens(info, x['ref'], side='target'),
            ),
            "sys_length": feature.Value(
                dtype="float",
                description="system output length",
                func=lambda info, x, c: count_tokens(info, x['ref'], side='target'),
            ),
            "src_divided_ref": feature.Value(
                dtype="float",
                description="ratio of source length to reference length",
                func=lambda info, x, c: c.features['src_length']
                / c.features['ref_length'],
            ),
            "sys_divided_ref": feature.Value(
                dtype="float",
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
        return self.continuous_feature_analyses()

    @classmethod
    def default_metrics(
        cls, level='example', source_language=None, target_language=None
    ) -> list[MetricConfig]:
        return [
            KtauCorrelationConfig(name='SegKtauCorr', group_by='segment'),
            PearsonCorrelationConfig(name='SysPearsonCorr', group_by='system'),
        ]

    # --- Feature functions accessible by ExplainaboardBuilder._get_feature_func()
    def _get_true_label(self, data_point: dict):
        """
        Get the true label from a data point. Returns "true_label" by default, but can
        be overloaded.
        :param data_point: the data point under consideration
        :return: the true label for the output
        """
        return [
            data_point["sys_name"],
            data_point["seg_id"],
            data_point["manual_raw"],
            data_point["manual_z"],
        ]

    def _get_predicted_label(self, data_point: dict):
        """
        Get the predicted label from a data point. Returns "predicted_label" by default,
        but can be overloaded.
        :param data_point: the data point under consideration
        :return: the predicted label for the output
        """
        return data_point["auto_score"]
