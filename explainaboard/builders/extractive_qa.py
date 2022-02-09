from dataclasses import dataclass, field, fields
from typing import Any, ClassVar, Dict, List, Optional
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from info import *
import feature
from typing import Iterable
from info import SysOutputInfo
from info import BucketPerformance
from info import Performance
from info import Table
from explainaboard.utils import analysis
from explainaboard.utils.analysis import *
from explainaboard.utils.eval_bucket import *
from explainaboard.utils.eval_basic_qa import *
from metric import Accuracy
from metric import F1score
from config import BuilderConfig
from tqdm import tqdm

from eaas import Config, Client
config = Config()
client = Client()
client.load_config(config)



""" TODO:
- [done] debug f1score metric for squad
- [ ] do we need a parent builder node?
- [ ] confidence interval
- [ ] metric class
- [ ] store sample_id instead of real examples
"""


class QASquadExplainaboardBuilder:

    def __init__(self,
                 info: SysOutputInfo,
                 system_output_object: Iterable[dict] = None,
                 feature_table: Optional[Table] = {},
                 gen_kwargs:dict = None
                 ):
        self._info = info
        self.gen_kwargs = gen_kwargs
        self._data: Table = feature_table
        self._system_output: Iterable[dict] = system_output_object
        # _samples_over_bucket_true: Dict(feature_name, bucket_name, sample_id_true_label):
        # samples in different buckets
        self._samples_over_bucket = {}
        # _performances_over_bucket: performance in different bucket: Dict(feature_name, bucket_name, performance)
        self._performances_over_bucket = {}



    @staticmethod
    def get_bucket_feature_value(feature_name:str):
        return "self._get_" + feature_name

    # define function for incomplete features
    def _get_context_length(self, existing_features: dict):
        return len(existing_features["context"].split(" "))

    def _get_question_length(self, existing_features: dict):
        return len(existing_features["question"].split(" "))


    def _get_answer_length(self, existing_features: dict):
        return len(existing_features["true_answers"]["text"][0].split(" "))


    def _get_sim_context_question(self, existing_features: dict):


        references = existing_features["context"]
        hypothesis = existing_features["question"]

        res_json = client.bleu([[references]], [hypothesis], lang = "en")
        return res_json["corpus_bleu"]





    def _complete_feature(self):
        """
        This function is used to calculate features used for bucekting, such as sentence_length
        :param feature_table_iterator:
        :return:
        """
        # Get names of bucketing features
        # print(f"self._info.features.get_bucket_features()\n {self._info.features.get_bucket_features()}")
        bucket_features = self._info.features.get_bucket_features()
        for _id, dict_sysout in tqdm(enumerate(self._system_output), desc="featurizing"):
            # Get values of bucketing features
            for bucket_feature in bucket_features:
                feature_value = eval(QASquadExplainaboardBuilder.get_bucket_feature_value(bucket_feature))(dict_sysout)
                dict_sysout[bucket_feature] = feature_value
            # if self._data == None:
            #     self._data = {}
            self._data[_id] = dict_sysout
            yield _id, dict_sysout

    def get_overall_performance(self):
        predicted_answers,true_answers = [], []

        for _id, feature_table in self._data.items():

            predicted_answers.append(feature_table["predicted_answer"])
            true_answers.append(feature_table["true_answers"]["text"])



        for metric_name in self._info.metric_names:
            overall_value = eval(metric_name)(true_answers,
                                           predicted_answers)




            overall_value = overall_value
            confidence_score_low = 0
            confidence_score_up = 0
            overall_performance = Performance(
                                   metric_name=metric_name,
                                   value=float(format(overall_value, '.4g')),
                                   confidence_score_low=float(format(confidence_score_low, '.4g')),
                                   confidence_score_up=float(format(confidence_score_up, '.4g')),
            )
            if self._info.results.overall == None:
                self._info.results.overall = {}
                self._info.results.overall[metric_name] = overall_performance
            else:
                self._info.results.overall[metric_name] = overall_performance


    def _bucketing_samples(self, sysout_iterator):

        sample_address= ""
        feature_to_sample_address_to_value = {}


        # Preparation for bucketing
        for _id, dict_sysout in sysout_iterator:
            # print(_id, dict_sysout)

            sample_address = str(_id) # this could be abstracted later
            for feature_name in self._info.features.get_bucket_features():
                if feature_name not in feature_to_sample_address_to_value.keys():
                    feature_to_sample_address_to_value[feature_name] = {}
                else:
                    feature_to_sample_address_to_value[feature_name][sample_address] = dict_sysout[feature_name]



        # Bucketing
        for feature_name in tqdm(self._info.features.get_bucket_features(), desc="bucketing"):

            # print(f"Feature Name: {feature_name}\n"
            #       f"Bucket Hyper:\n function_name: {self._info.features[feature_name].bucket_info._method} \n"
            #       f"bucket_number: {self._info.features[feature_name].bucket_info._number}\n"
            #       f"bucket_setting: {self._info.features[feature_name].bucket_info._setting}\n")

            self._samples_over_bucket[feature_name] = eval(self._info.features[feature_name].bucket_info._method)(
                                dict_obj = feature_to_sample_address_to_value[feature_name],
                                bucket_number = self._info.features[feature_name].bucket_info._number,
                                bucket_setting = self._info.features[feature_name].bucket_info._setting)

            # print(f"self._samples_over_bucket.keys():\n{self._samples_over_bucket.keys()}")

            # evaluating bucket: get bucket performance
            self._performances_over_bucket[feature_name] = self.get_bucket_performance(feature_name)

    def get_bucket_performance(self, feature_name:str):
        """
        This function defines how to get bucket-level performance w.r.t a given feature (e.g., sentence length)
        :param feature_name: the name of a feature, e.g., sentence length
        :return: bucket_name_to_performance: a dictionary that maps bucket names to bucket performance
        """

        bucket_name_to_performance = {}
        for bucket_interval, sample_ids in self._samples_over_bucket[feature_name].items():

            bucket_true_labels      = []
            bucket_predicted_labels = []
            bucket_cases = []


            for sample_id in sample_ids:

                true_label = self._data[int(sample_id)]["true_answers"]["text"]

                predicted_label = self._data[int(sample_id)]["predicted_answer"]
                sent = self._data[int(sample_id)]["question"]

                # get a bucket of true/predicted labels
                bucket_true_labels.append(true_label)
                bucket_predicted_labels.append(predicted_label)
                # get a bucket of cases (e.g., errors)
                if self._info.results.is_print_case:
                    if true_label[0] != predicted_label:
                        #bucket_case = true_label[0] + "|||" + predicted_label + "|||" + sent
                        # bucket_case = {"true_answer": (sample_id, ["true_answers","text"]),
                        #                "predicted_answer": (sample_id, ["predicted_answer"]),
                        #                "question": (sample_id, ["question"])}
                        system_output_id = self._data[int(sample_id)]["id"]
                        bucket_case = system_output_id
                        bucket_cases.append(bucket_case)

            bucket_name_to_performance[bucket_interval] = []
            for metric_name in self._info.metric_names:

                bucket_value = eval(metric_name)(bucket_true_labels,
                                           bucket_predicted_labels)



                bucket_value = bucket_value
                confidence_score_low = 0
                confidence_score_up = 0


                # print(
                #       f"value:\t {bucket_value}\n"
                #       f"confidence low\t {confidence_score_low}\n"
                #       f"confidence up \t {confidence_score_up}\n"
                #       f"---------------------------------")

                bucket_performance = BucketPerformance(bucket_name=bucket_interval,
                                  metric_name = metric_name,
                                  value = format(bucket_value, '.4g'),
                                  confidence_score_low = format(confidence_score_low, '.4g'),
                                  confidence_score_up = format(confidence_score_up, '.4g'),
                                  n_samples = len(bucket_true_labels),
                                  bucket_samples=bucket_cases)

                bucket_name_to_performance[bucket_interval].append(bucket_performance)




        return sort_dict(bucket_name_to_performance)


    def _generate_report(self):

        dict_fine_grained = {}
        for feature_name, metadata in self._performances_over_bucket.items():
            dict_fine_grained[feature_name] = []
            for bucket_name, bucket_performance in metadata.items():
                bucket_name = analysis.beautify_interval(bucket_name)

                # instantiation
                dict_fine_grained[feature_name].append(bucket_performance)

        self._info.results.fine_grained = dict_fine_grained
        # self._info.write_to_directory(file_directory)

    def _print_bucket_info(self):
        for feature_name in self._performances_over_bucket.keys():
            print_dict(self._performances_over_bucket[feature_name], feature_name)


    def run(self):

        eb_generator = self._complete_feature()
        self._bucketing_samples(eb_generator)
        self.get_overall_performance()
        self._print_bucket_info()
        self._generate_report()
        return self._info



