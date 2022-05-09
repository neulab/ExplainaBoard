from __future__ import annotations

from collections.abc import Iterator
import json

from datalabs import aggregating

from explainaboard import feature, TaskType
from explainaboard.info import SysOutputInfo
from explainaboard.metric import (
    Hits,
    HitsConfig,
    MeanRank,
    MeanRankConfig,
    MeanReciprocalRank,
    MeanReciprocalRankConfig,
    MetricConfig,
    MetricStats,
)
from explainaboard.processors.processor import Processor
from explainaboard.processors.processor_registry import register_processor
from explainaboard.utils import cache_api
from explainaboard.utils.logging import progress
from explainaboard.utils.typing_utils import unwrap


@register_processor(TaskType.kg_link_tail_prediction)
class KGLinkTailPredictionProcessor(Processor):
    @classmethod
    def task_type(cls) -> TaskType:
        return TaskType.kg_link_tail_prediction

    @classmethod
    def default_features(cls) -> feature.Features:
        return feature.Features(
            {
                "true_head": feature.Value("string"),
                "true_head_decipher": feature.Value("string"),
                "true_link": feature.Value("string"),
                "true_tail": feature.Value("string"),
                "true_tail_decipher": feature.Value("string"),
                "predict": feature.Value("string"),
                "true_label": feature.Value("string"),
                "predictions": feature.Sequence(feature=feature.Value("string")),
                "tail_entity_length": feature.Value(
                    dtype="float",
                    description="number of words in the tail entity",
                    is_bucket=True,
                    bucket_info=feature.BucketInfo(
                        method="bucket_attribute_specified_bucket_value",
                        number=4,
                        setting=(),
                    ),
                ),
                "head_entity_length": feature.Value(
                    dtype="float",
                    description="number of words in the head entity",
                    is_bucket=True,
                    bucket_info=feature.BucketInfo(
                        method="bucket_attribute_specified_bucket_value",
                        number=4,
                        setting=(),
                    ),
                ),
                "tail_fre": feature.Value(
                    dtype="float",
                    description="the frequency of tail entity in the training set",
                    is_bucket=True,
                    bucket_info=feature.BucketInfo(
                        method="bucket_attribute_specified_bucket_value",
                        number=4,
                        setting=(),
                    ),
                    require_training_set=True,
                ),
                "link_fre": feature.Value(
                    dtype="float",
                    description="the frequency of link relation in the training set",
                    is_bucket=True,
                    bucket_info=feature.BucketInfo(
                        method="bucket_attribute_specified_bucket_value",
                        number=4,
                        setting=(),
                    ),
                    require_training_set=True,
                ),
                "head_fre": feature.Value(
                    dtype="float",
                    description="the frequency of head relation in the training set",
                    is_bucket=True,
                    bucket_info=feature.BucketInfo(
                        method="bucket_attribute_specified_bucket_value",
                        number=4,
                        setting=(),
                    ),
                    require_training_set=True,
                ),
                "symmetry": feature.Value(
                    dtype="string",
                    description=(
                        "boolean feature: 'symmetric' or 'asymmetric'; more "
                        "granularity to be added"
                    ),
                    is_bucket=True,
                    bucket_info=feature.BucketInfo(
                        method="bucket_attribute_discrete_value", number=2, setting=1
                    ),
                ),
                "entity_type_level": feature.Value(
                    dtype="string",
                    description=(
                        "most specific (highest) entity type level of true tail entity"
                    ),
                    is_bucket=True,
                    bucket_info=feature.BucketInfo(
                        method="bucket_attribute_discrete_value", number=8, setting=1
                    ),
                ),
            }
        )

    @classmethod
    def default_metrics(
        cls, source_language=None, target_language=None
    ) -> list[MetricConfig]:
        return [
            HitsConfig(name='Hits1', hits_k=1),
            HitsConfig(name='Hits2', hits_k=2),
            HitsConfig(name='Hits3', hits_k=3),
            HitsConfig(name='Hits5', hits_k=5),
            MeanReciprocalRankConfig(name='MRR'),
            MeanRankConfig(name='MR'),
        ]

    # TODO: is this the best place to put this?
    _symmetric_relations = [
        '/base/popstra/celebrity/breakup./base/popstra/breakup/participant',
        '/base/popstra/celebrity/canoodled./base/popstra/canoodled/participant',
        '/base/popstra/celebrity/dated./base/popstra/dated/participant',
        '/base/popstra/celebrity/friendship./base/popstra/friendship/participant',
        '/celebrities/celebrity/celebrity_friends./celebrities/friendship/friend',
        '/celebrities/celebrity/sexual_relationships./celebrities/romantic_relationship/celebrity',  # noqa: E501
        '/influence/influence_node/peers./influence/peer_relationship/peers',
        '/location/location/adjoin_s./location/adjoining_relationship/adjoins',
        '/people/person/spouse_s./people/marriage/spouse',
        '/people/person/sibling_s./people/sibling relationship/sibling',
    ]

    def __init__(self):
        super().__init__()
        self.entity_type_level_map = None
        file_path = cache_api.cache_online_file(
            'http://phontron.com/download/explainaboard/pre_computed/kg/entity_type_level_map.json',  # noqa
            'pre_computed/kg/entity_type_level_map.json',
        )
        with open(file_path, 'r') as file:
            self.entity_type_level_map = json.load(file)

    @aggregating()
    def _statistics_func(self, samples: Iterator, sys_info: SysOutputInfo):
        """
        `Samples` is a dataset iterator: List[Dict], to know more about it, you can:
        # pip install datalabs
        dataset = load_dataset("fb15k_237", 'readable')
        print(dataset['train'])
        """
        dict_head: dict[str, int] = {}
        dict_link: dict[str, int] = {}
        dict_tail: dict[str, int] = {}

        entity_dic = {}
        file_path = cache_api.cache_online_file(
            'http://phontron.com/download/explainaboard/pre_computed/kg/entity2wikidata.json',  # noqa
            'pre_computed/kg/entity2wikidata.json',
        )
        with open(file_path, 'r') as file:
            entity_dic = json.loads(file.read())

        for sample in progress(samples):

            tail = (
                sample['tail']
                if sample['tail'] not in entity_dic.keys()
                else entity_dic[sample['tail']]['label']
            )
            if tail not in dict_tail.keys():
                dict_tail[tail] = 1
            else:
                dict_tail[tail] += 1

            head = (
                sample['head']
                if sample['head'] not in entity_dic.keys()
                else entity_dic[sample['head']]['label']
            )
            if head not in dict_head.keys():
                dict_head[head] = 1
            else:
                dict_head[head] += 1

            link = (
                sample['link']
                if sample['link'] not in entity_dic.keys()
                else entity_dic[sample['link']]['label']
            )
            if link not in dict_link.keys():
                dict_link[link] = 1
            else:
                dict_link[link] += 1

        return {
            "head_fre": dict_head,
            "link_fre": dict_link,
            "tail_fre": dict_tail,
        }

    def _gen_metric_stats(
        self, sys_info: SysOutputInfo, sys_output: list[dict]
    ) -> list[MetricStats]:
        """Generate sufficient statistics for scoring different metrics.
        :param sys_info: Information about the system outputs
        :param sys_output: The system output itself
        :return: Statistics sufficient for scoring
        """

        metrics = unwrap(self._get_metrics(sys_info))
        true_data = [self._get_true_label(x) for x in sys_output]
        pred_data = [self._get_predicted_label(x) for x in sys_output]
        rank_data = [
            self._get_rank_data(x) for x in sys_output
        ]  # rank of true entity in predictions

        if any(item is None for item in rank_data):
            raise ValueError(
                'Some data points do not have rank information; check system outputs.'
            )

        metric_stats = []
        for metric in metrics:
            if (
                isinstance(metric, MeanReciprocalRank)
                or isinstance(metric, MeanRank)
                or isinstance(metric, Hits)
            ):
                metric_stats.append(metric.calc_stats_from_rank(rank_data))
            else:
                metric_stats.append(metric.calc_stats_from_data(true_data, pred_data))
        return metric_stats

    # --- Feature functions accessible by ExplainaboardBuilder._get_feature_func()
    def _get_entity_type_level(self, sys_info: SysOutputInfo, existing_features: dict):

        # list of entity types at each level:
        # [type_level_0, type_level_1, ... type_level_6]
        # e.g. ["Thing", "Agent", "Person", None, None, None, None]
        tail_entity_type_levels = self.entity_type_level_map.get(
            existing_features['true_tail'], None
        )
        if tail_entity_type_levels is None:
            return "-1"  # entity types not found

        # find the index of the first occurrence of None in the list
        if None in tail_entity_type_levels:
            most_specific_level = tail_entity_type_levels.index(None) - 1
        else:  # tail has entity types at every level
            most_specific_level = len(tail_entity_type_levels) - 1
        return str(most_specific_level)

    def _get_tail_entity_length(self, sys_info: SysOutputInfo, existing_features: dict):
        return len(
            unwrap(sys_info.target_tokenizer)(existing_features["true_tail_decipher"])
        )

    def _get_head_entity_length(self, sys_info: SysOutputInfo, existing_features: dict):
        return len(
            unwrap(sys_info.source_tokenizer)(existing_features["true_head_decipher"])
        )

    def _get_tail_fre(
        self, sys_info: SysOutputInfo, existing_features: dict, statistics
    ):
        if (
            statistics is None
            or existing_features["true_tail_decipher"]
            not in statistics['tail_fre'].keys()
        ):
            return 0
        else:
            return statistics['tail_fre'][existing_features["true_tail_decipher"]]

    def _get_head_fre(
        self, sys_info: SysOutputInfo, existing_features: dict, statistics
    ):
        if (
            statistics is None
            or existing_features["true_head_decipher"]
            not in statistics['head_fre'].keys()
        ):
            return 0
        else:
            return statistics['head_fre'][existing_features["true_head_decipher"]]

    def _get_link_fre(
        self, sys_info: SysOutputInfo, existing_features: dict, statistics
    ):
        if (
            statistics is None
            or existing_features["true_link"] not in statistics['link_fre'].keys()
        ):
            return 0
        else:
            return statistics['link_fre'][existing_features["true_link"]]

    def _get_symmetry(self, sys_info: SysOutputInfo, existing_features: dict):
        if existing_features['true_link'] in self._symmetric_relations:
            return 'symmetric'
        else:
            return 'asymmetric'

    # --- End feature functions

    def _get_true_label(self, data_point: dict):
        return data_point["true_" + data_point["predict"]]

    def _get_predicted_label(self, data_point: dict):
        return data_point["predictions"]

    def _get_rank_data(self, data_point: dict):
        if "true_rank" in data_point.keys():
            return data_point["true_rank"]
        else:
            return None
