from typing import Iterable, Optional
from explainaboard.info import SysOutputInfo, BucketPerformance, Performance, Table
from explainaboard.utils import analysis
from explainaboard.builders import ExplainaboardBuilder
from explainaboard.utils.eval_bucket import *  # noqa
from explainaboard.utils.analysis import *  # noqa
from explainaboard.metric import *  # noqa
from tqdm import tqdm
from datalabs import load_dataset
from datalabs.operations import aggregate
from datalabs.operations.aggregate.kg_link_prediction import (
    kg_link_prediction_aggregating,
)
from typing import Iterator, Dict, List
import sys
import os


@kg_link_prediction_aggregating(
    name="get_statistics",
    contributor="datalab",
    task="kg-link-prediction",
    description="aggregation function",
)
def get_statistics(samples: List[Dict]):
    """
    `Samples` is a dataset iterator: List[Dict], to know more about it, you can:
    # pip install datalabs
    dataset = load_dataset("fb15k_237", 'readable')
    print(dataset['train'])
    """
    dict_head = {}
    dict_link = {}
    dict_tail = {}

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


# TODO: is this the best place to put this?
SYMMETRIC_RELATIONS = [
    '/base/popstra/celebrity/breakup./base/popstra/breakup/participant',
    '/base/popstra/celebrity/canoodled./base/popstra/canoodled/participant',
    '/base/popstra/celebrity/dated./base/popstra/dated/participant',
    '/base/popstra/celebrity/friendship./base/popstra/friendship/participant',
    '/celebrities/celebrity/celebrity_friends./celebrities/friendship/friend',
    '/celebrities/celebrity/sexual_relationships./celebrities/romantic_relationship/celebrity',
    '/influence/influence_node/peers./influence/peer_relationship/peers',
    '/location/location/adjoin_s./location/adjoining_relationship/adjoins',
    '/people/person/spouse_s./people/marriage/spouse',
    '/people/person/sibling_s./people/sibling relationship/sibling',
]


class KGLTPExplainaboardBuilder(ExplainaboardBuilder):
    """
    Input: System Output file List[dict];  Metadata info
    Output: Analysis
    """

    def __init__(
        self,
        info: SysOutputInfo,
        system_output_object: Iterable[dict],
        feature_table: Optional[Table] = None,
        user_defined_feature_config=None,
    ):
        super().__init__(
            info, system_output_object, feature_table, user_defined_feature_config
        )

        # TODO(gneubig): this should be deduplicated
        # Calculate statistics of training set
        self.statistics = None
        if None != self._info.dataset_name:
            try:
                dataset = load_dataset(self._info.dataset_name, "readable")
                if (
                    len(dataset['train']._stat) == 0 or self._info.reload_stat == False
                ):  # calculate the statistics (_stat) when _stat is {} or `reload_stat` is False
                    new_train = dataset['train'].apply(get_statistics, mode="local")
                    self.statistics = new_train._stat
                else:
                    self.statistics = dataset["train"]._stat
            except FileNotFoundError as err:
                eprint(
                    "The dataset hasn't been supported by DataLab so no training set dependent features will be supported by ExplainaBoard."
                    "You can add the dataset by: https://github.com/ExpressAI/DataLab/blob/main/docs/SDK/add_new_datasets_into_sdk.md"
                )

        # entity types
        if self._info.dataset_name != "fb15k_237":  # to be generalized
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
        # print(self.entity_type_level_map)
        # exit()

        # f = open('entity_type_level_map.json')
        # self.entity_type_level_map = json.load(f)
        # print(self.entity_type_level_map.keys())

    # --- Feature functions accessible by ExplainaboardBuilder._get_feature_func()
    def _get_entity_type_level(self, existing_features: dict):

        # list of entity types at each level: [type_level_0, type_level_1, ... type_level_6]
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
        return len(existing_features["true_tail"].split(" "))

    def _get_head_entity_length(self, existing_features: dict):
        return len(existing_features["true_head"].split(" "))

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
        if existing_features['relation'] in SYMMETRIC_RELATIONS:
            return 'symmetric'
        else:
            return 'asymmetric'

    # --- End feature functions

    def _complete_feature(self):
        """
        This function is used to calculate features used for bucekting, such as sentence_length
        :param feature_table_iterator:
        :return:
        """
        # Get names of bucketing features
        # print(f"self._info.features.get_bucket_features()\n {self._info.features.get_bucket_features()}")
        bucket_features = self._info.features.get_bucket_features()
        for _id, dict_sysout in tqdm(
            enumerate(self._system_output), desc="featurizing"
        ):
            # Get values of bucketing features
            for bucket_feature in bucket_features:
                if (
                    self._user_defined_feature_config is not None
                    and bucket_feature in self._user_defined_feature_config.keys()
                ):
                    feature_value = dict_sysout[bucket_feature]
                # this is needed due to `del self._info.features[bucket_feature]`
                elif bucket_feature in self._info.features.keys():
                    # If there is a training set dependent feature while no pre-computed statistics for it,
                    # then skip bucketing along this feature
                    if (
                        self._info.features[bucket_feature].require_training_set
                        and self.statistics == None
                    ):
                        del self._info.features[bucket_feature]
                    else:
                        feature_value = self._get_feature_func(bucket_feature)(
                            dict_sysout
                        )
                dict_sysout[bucket_feature] = feature_value
            # if self._data is None:
            #     self._data = {}
            self._data[_id] = dict_sysout
            yield _id, dict_sysout

    # TODO(gneubig): should this be generalized or is it task specific?
    def get_overall_performance(self):
        predicted_labels, true_labels = [], []

        for _id, feature_table in self._data.items():

            predicted_labels.append(feature_table["predicted_tails"])
            true_labels.append(feature_table["true_tail"])

        for metric_name in self._info.metric_names:
            one_metric = eval(metric_name)(
                true_labels=true_labels,
                predicted_labels=predicted_labels,
                is_print_confidence_interval=self._info.results.is_print_confidence_interval,
            )
            overall_value_json = one_metric.evaluate()

            overall_value = overall_value_json["value"]
            confidence_score_low = overall_value_json["confidence_score_low"]
            confidence_score_up = overall_value_json["confidence_score_up"]
            overall_performance = Performance(
                metric_name=metric_name,
                value=float(format(overall_value, '.4g')),
                confidence_score_low=float(format(confidence_score_low, '.4g')),
                confidence_score_up=float(format(confidence_score_up, '.4g')),
            )

            if self._info.results.overall is None:
                self._info.results.overall = {}
                self._info.results.overall[metric_name] = overall_performance
            else:
                self._info.results.overall[metric_name] = overall_performance

    def get_bucket_performance(self, feature_name: str):
        """
        This function defines how to get bucket-level performance w.r.t a given feature (e.g., sentence length)
        :param feature_name: the name of a feature, e.g., sentence length
        :return: bucket_name_to_performance: a dictionary that maps bucket names to bucket performance
        """

        bucket_name_to_performance = {}
        for bucket_interval, sample_ids in self._samples_over_bucket[
            feature_name
        ].items():

            bucket_true_labels = []
            bucket_predicted_labels = []  # list of (lists of top-k ranked tails)
            bucket_cases = []

            for sample_id in sample_ids:

                true_label = self._data[int(sample_id)]["true_tail"]
                predicted_label = self._data[int(sample_id)]["predicted_tails"]
                sent = self._data[int(sample_id)]["link"]  # noqa
                s_id = self._data[int(sample_id)]["id"]

                # get a bucket of true/predicted labels
                bucket_true_labels.append(true_label)
                bucket_predicted_labels.append(predicted_label)
                # get a bucket of cases (e.g., errors)
                if self._info.results.is_print_case:
                    if true_label not in predicted_label:
                        # bucket_case = true_label + "|||" + predicted_label + "|||" + sent
                        # bucket_case = {"true_label":(s_id,["true_label"]),
                        #                "predicted_label":(s_id,["predicted_label"]),
                        #                "text":(s_id,["text"])}
                        bucket_case = str(s_id)
                        bucket_cases.append(bucket_case)

            bucket_name_to_performance[bucket_interval] = []
            for metric_name in self._info.metric_names:

                one_metric = eval(metric_name)(
                    true_labels=bucket_true_labels,
                    predicted_labels=bucket_predicted_labels,
                    is_print_confidence_interval=self._info.results.is_print_confidence_interval,
                )
                bucket_value_json = one_metric.evaluate()

                bucket_value = bucket_value_json["value"]
                confidence_score_low = bucket_value_json["confidence_score_low"]
                confidence_score_up = bucket_value_json["confidence_score_up"]

                # print(f"name:\t {one_metric._name} \n"
                #       f"value:\t {bucket_value}\n"
                #       f"confidence low\t {confidence_score_low}\n"
                #       f"confidence up \t {confidence_score_up}\n"
                #       f"---------------------------------")

                bucket_performance = BucketPerformance(
                    bucket_name=bucket_interval,
                    metric_name=metric_name,
                    value=format(bucket_value, '.4g'),
                    confidence_score_low=format(confidence_score_low, '.4g'),
                    confidence_score_up=format(confidence_score_up, '.4g'),
                    n_samples=len(bucket_true_labels),
                    bucket_samples=bucket_cases,
                )

                bucket_name_to_performance[bucket_interval].append(bucket_performance)

        return sort_dict(bucket_name_to_performance)  # noqa
