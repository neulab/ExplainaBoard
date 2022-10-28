"""A processor for the knowledge graph link tail prediction task."""

from __future__ import annotations

from collections.abc import Iterable
import json
from typing import Any

from explainaboard import TaskType
from explainaboard.analysis import feature
from explainaboard.analysis.analyses import Analysis, AnalysisLevel, BucketAnalysis
from explainaboard.analysis.case import AnalysisCase
from explainaboard.analysis.feature_funcs import count_tokens
from explainaboard.info import SysOutputInfo
from explainaboard.metrics.metric import MetricConfig, MetricStats
from explainaboard.metrics.ranking import (
    HitsConfig,
    MeanRankConfig,
    MeanReciprocalRankConfig,
    RankingMetric,
)
from explainaboard.processors.processor import Processor
from explainaboard.utils import cache_api
from explainaboard.utils.logging import progress
from explainaboard.utils.typing_utils import narrow


class KGLinkTailPredictionProcessor(Processor):
    """A processor for the knowledge graph link tail prediction task."""

    @classmethod
    def task_type(cls) -> TaskType:
        """See Processor.task_type."""
        return TaskType.kg_link_tail_prediction

    def default_analysis_levels(self) -> list[AnalysisLevel]:
        """See Processor.default_analysis_levels."""
        features = {
            "true_head": feature.Value(dtype=feature.DataType.STRING),
            "true_head_decipher": feature.Value(dtype=feature.DataType.STRING),
            "true_link": feature.Value(
                dtype=feature.DataType.STRING, description="the relation type"
            ),
            "true_tail": feature.Value(dtype=feature.DataType.STRING),
            "true_tail_decipher": feature.Value(dtype=feature.DataType.STRING),
            "predict": feature.Value(dtype=feature.DataType.STRING),
            "predictions": feature.Sequence(
                feature=feature.Value(dtype=feature.DataType.STRING)
            ),
            "tail_entity_length": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="length of the tail entity in tokens",
                func=lambda info, x, c: count_tokens(
                    info, x["true_tail_decipher"], side="target"
                ),
            ),
            "head_entity_length": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="length of the head entity in tokens",
                func=lambda info, x, c: count_tokens(
                    info, x["true_head_decipher"], side="target"
                ),
            ),
            "tail_fre": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="average frequency of the tail entity",
                require_training_set=True,
                func=lambda info, x, c, stat: stat["tail_fre"].get(
                    x["true_tail_decipher"], 0
                ),
            ),
            "link_fre": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="frequency of relation in training set",
                require_training_set=True,
                func=lambda info, x, c, stat: stat["link_fre"].get(x["true_link"], 0),
            ),
            "head_fre": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="frequency of head entity in training set",
                require_training_set=True,
                func=lambda info, x, c, stat: stat["head_fre"].get(
                    x["true_head_decipher"], 0
                ),
            ),
            "symmetry": feature.Value(
                dtype=feature.DataType.STRING,
                description="whether the relation is symmetric",
                func=lambda info, x, c: "symmetric"
                if x["true_link"] in self._symmetric_relations
                else "asymmetric",
            ),
            "entity_type_level": feature.Value(
                dtype=feature.DataType.STRING,
                description="most specific entity type level of the true tail entity",
                func=lambda info, x, c: self._get_entity_type_level(x),
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
        analysis_levels = self.default_analysis_levels()
        features = analysis_levels[0].features
        discrete_features = {"symmetry": 2, "entity_type_level": 8, "true_link": 15}
        analyses: list[Analysis] = [
            BucketAnalysis(
                level=analysis_levels[0].name,
                description=features[k].description,
                feature=k,
                method="discrete",
                num_buckets=v,
            )
            for k, v in discrete_features.items()
        ]
        analyses.extend(self.continuous_feature_analyses())
        return analyses

    @classmethod
    def default_metrics(
        cls,
        level: str = "example",
        source_language: str | None = None,
        target_language: str | None = None,
    ) -> dict[str, MetricConfig]:
        """See Processor.default_metrics."""
        return {
            "Hits1": HitsConfig(hits_k=1),
            "Hits2": HitsConfig(hits_k=2),
            "Hits3": HitsConfig(hits_k=3),
            "Hits5": HitsConfig(hits_k=5),
            "Hits10": HitsConfig(hits_k=10),
            "MRR": MeanReciprocalRankConfig(),
            "MR": MeanRankConfig(),
        }

    # TODO: is this the best place to put this?
    _symmetric_relations = {
        "/base/popstra/celebrity/breakup./base/popstra/breakup/participant",
        "/base/popstra/celebrity/canoodled./base/popstra/canoodled/participant",
        "/base/popstra/celebrity/dated./base/popstra/dated/participant",
        "/base/popstra/celebrity/friendship./base/popstra/friendship/participant",
        "/celebrities/celebrity/celebrity_friends./celebrities/friendship/friend",
        "/celebrities/celebrity/sexual_relationships./celebrities/romantic_relationship/celebrity",  # noqa: E501
        "/influence/influence_node/peers./influence/peer_relationship/peers",
        "/location/location/adjoin_s./location/adjoining_relationship/adjoins",
        "/people/person/spouse_s./people/marriage/spouse",
        "/people/person/sibling_s./people/sibling relationship/sibling",
    }

    def __init__(self) -> None:
        """Constructor."""
        super().__init__()
        file_path = cache_api.cache_online_file(
            "https://storage.googleapis.com/inspired-public-data/"
            "explainaboard/task_data/kg_link_tail_prediction/entity2wikidata.json",
            "explainaboard/task_data/kg_link_tail_prediction/entity2wikidata.json",
        )
        with open(file_path, "r") as file:
            self.entity_type_level_map: dict = json.load(file)

    def _statistics_func(self, samples: Iterable[Any], sys_info: SysOutputInfo) -> dict:
        """See Processor._statistics_func."""
        dict_head: dict[str, int] = {}
        dict_link: dict[str, int] = {}
        dict_tail: dict[str, int] = {}

        for sample in progress(samples):

            tail = sample["true_tail_decipher"]
            dict_tail[tail] = dict_tail.get(tail, 0) + 1

            head = sample["true_head_decipher"]
            dict_head[head] = dict_head.get(head, 0) + 1

            link = sample["true_link"]
            dict_link[link] = dict_link.get(link, 0) + 1

        return {
            "head_fre": dict_head,
            "link_fre": dict_link,
            "tail_fre": dict_tail,
        }

    def _gen_cases_and_stats(
        self,
        sys_info: SysOutputInfo,
        sys_output: list[dict],
        statistics: Any,
        analysis_level: AnalysisLevel,
    ) -> tuple[list[AnalysisCase], dict[str, MetricStats]]:
        # Note that this is overridden to calculate stats from rank
        cases = []
        true_data = [self._get_true_label(x) for x in sys_output]
        pred_data = [self._get_predicted_label(x) for x in sys_output]
        rank_data = [narrow(int, x.get("true_rank")) for x in sys_output]
        if any(item is None for item in rank_data):
            raise ValueError(
                "Some data points do not have rank information; check system outputs."
            )

        metric_stats: dict[str, MetricStats] = {}
        for name, config in analysis_level.metric_configs.items():
            metric = config.to_metric()
            if isinstance(metric, RankingMetric):
                metric_stats[name] = metric.calc_stats_from_rank(rank_data)
            else:
                metric_stats[name] = metric.calc_stats_from_data(true_data, pred_data)

        # Calculate features
        for i, output in progress(
            enumerate(sys_output), desc="calculating example-level features"
        ):
            case = AnalysisCase(sample_id=i, features={})
            for feat_name, feat_spec in analysis_level.features.items():
                if feat_spec.func is None:
                    case.features[feat_name] = output[feat_name]
                elif not feat_spec.require_training_set:
                    case.features[feat_name] = feat_spec.func(sys_info, output, case)
                elif statistics is not None:
                    case.features[feat_name] = feat_spec.func(
                        sys_info, output, case, statistics
                    )
            cases.append(case)
        return cases, metric_stats

    # --- Feature functions accessible by ExplainaboardBuilder._get_feature_func()
    def _get_entity_type_level(self, existing_features: dict):

        # entities not found in `entity_type_level_map` get bucketed to this value.
        # in FB15k, "0" is the same as the most generic entity type, "Thing".
        default_level = "0"

        # entities not found in `entity_type_level_map` get bucketed to this value.
        # in FB15k, "0" is the same as the most generic entity type, "Thing".
        default_level = "0"

        # list of entity types at each level:
        # [type_level_0, type_level_1, ... type_level_6]
        # e.g. ["Thing", "Agent", "Person", None, None, None, None]
        tail_entity_type_levels = self.entity_type_level_map.get(
            existing_features["true_tail"], None
        )
        if tail_entity_type_levels is None:
            return default_level  # entity types not found

        # find the index of the first occurrence of None in the list
        if None in tail_entity_type_levels:
            most_specific_level = tail_entity_type_levels.index(None) - 1
        else:  # tail has entity types at every level
            most_specific_level = len(tail_entity_type_levels) - 1
        return str(most_specific_level)

    # --- End feature functions

    def _get_true_label(self, data_point: dict):
        """See processor._get_true_label."""
        return data_point["true_" + data_point["predict"]]

    def _get_predicted_label(self, data_point: dict):
        """See processor._get_predicted_label."""
        return data_point["predictions"]
