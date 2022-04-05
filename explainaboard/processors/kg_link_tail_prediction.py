from __future__ import annotations

from collections.abc import Callable, Iterator
import json
import os

from datalabs import aggregating, load_dataset
from tqdm import tqdm

# TODO(odashi): Add a function to obtain metric class instead of using getattr.
from explainaboard import feature
from explainaboard.info import SysOutputInfo
from explainaboard.processors.processor import Processor
from explainaboard.processors.processor_registry import register_processor
from explainaboard.tasks import TaskType
from explainaboard.utils.py_utils import eprint


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
                "link": feature.Value("string"),
                "true_tail": feature.Value("string"),
                "predicted_tails": feature.Sequence(feature.Value("string")),
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
    def default_metrics(cls) -> list[str]:
        return ["Hits", "MeanReciprocalRank"]

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
        self._user_defined_feature_config = None

    @aggregating()
    def _statistics_func(self, samples: Iterator[dict[str, str]]):
        """
        `Samples` is a dataset iterator: List[Dict], to know more about it, you can:
        # pip install datalabs
        dataset = load_dataset("fb15k_237", 'readable')
        print(dataset['train'])
        """
        dict_head: dict[str, int] = {}
        dict_link: dict[str, int] = {}
        dict_tail: dict[str, int] = {}

        for sample in tqdm(samples):

            if sample['tail'] not in dict_tail.keys():
                dict_tail[sample['tail']] = 1
            else:
                dict_tail[sample['tail']] += 1

            if sample['head'] not in dict_head.keys():
                dict_head[sample['head']] = 1
            else:
                dict_head[sample['head']] += 1

            if sample['link'] not in dict_link.keys():
                dict_link[sample['link']] = 1
            else:
                dict_link[sample['link']] += 1

        return {
            "head_fre": dict_head,
            "link_fre": dict_link,
            "tail_fre": dict_tail,
        }

    def _gen_external_stats(self, sys_info: SysOutputInfo, statistics_func: Callable):

        # TODO(gneubig):
        # this will be reloaded for every dataset, maybe should be fixed for multiple
        # analysis
        if sys_info.dataset_name != "fb15k_237":  # to be generalized
            self.entity_type_level_map = {}
        else:
            scriptpath = os.path.dirname(__file__)
            with open(
                os.path.join(
                    scriptpath, '../pre_computed/kg/entity_type_level_map.json'
                ),
                'r',
            ) as file:
                self.entity_type_level_map = json.loads(file.read())

        # Calculate statistics of training set
        self.statistics = None
        if sys_info.dataset_name is not None:
            try:
                dataset = load_dataset(sys_info.dataset_name, "readable")
                # calculate the statistics (_stat) when _stat is {} or
                # `reload_stat` is False
                if len(dataset['train']._stat) == 0 or not sys_info.reload_stat:
                    new_train = dataset['train'].apply(
                        self._statistics_func, mode="local"
                    )
                    self.statistics = new_train._stat
                else:
                    self.statistics = dataset["train"]._stat
            except FileNotFoundError:
                eprint(
                    """
The dataset hasn't been supported by DataLab so no training set dependent features will
be supported by ExplainaBoard. You can add the dataset by:
https://github.com/ExpressAI/DataLab/blob/main/docs/SDK/add_new_datasets_into_sdk.md
"""
                )

    # --- Feature functions accessible by ExplainaboardBuilder._get_feature_func()
    def _get_entity_type_level(self, existing_features: dict):

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

    def _get_tail_entity_length(self, existing_features: dict):
        return len(self._tokenizer(existing_features["true_tail"]))

    def _get_head_entity_length(self, existing_features: dict):
        return len(self._tokenizer(existing_features["true_head"]))

    def _get_tail_fre(self, existing_features: dict):
        if (
            self.statistics is None
            or existing_features["true_tail"] not in self.statistics['tail_fre'].keys()
        ):
            return 0
        else:
            return self.statistics['tail_fre'][existing_features["true_tail"]]

    def _get_head_fre(self, existing_features: dict):
        if (
            self.statistics is None
            or existing_features["true_head"] not in self.statistics['head_fre'].keys()
        ):
            return 0
        else:
            return self.statistics['head_fre'][existing_features["true_head"]]

    def _get_link_fre(self, existing_features: dict):
        if (
            self.statistics is None
            or existing_features["link"] not in self.statistics['link_fre'].keys()
        ):
            return 0
        else:
            return self.statistics['link_fre'][existing_features["link"]]

    def _get_symmetry(self, existing_features: dict):
        if existing_features['relation'] in self._symmetric_relations:
            return 'symmetric'
        else:
            return 'asymmetric'

    # --- End feature functions

    def _get_true_label(self, data_point: dict):
        return data_point["true_tail"]

    def _get_predicted_label(self, data_point: dict):
        return data_point["predicted_tails"]
