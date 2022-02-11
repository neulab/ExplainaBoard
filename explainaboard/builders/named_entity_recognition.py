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
from explainaboard.utils.analysis import *
from explainaboard.utils.eval_basic import *
from explainaboard.utils.eval_bucket import *
from metric import Accuracy
from metric import F1score
from config import BuilderConfig
import pickle
from tqdm import tqdm

"""
- [ ] Automatically delete features without pre-trained dict
- [ ] Tag features
"""


class NERExplainaboardBuilder:

    def __init__(self, info: SysOutputInfo,
                 system_output_object: Iterable[dict],
                 feature_table: Optional[Table] = {},
                 gen_kwargs:dict = None
                 ):
        self._info = info
        self._system_output: Iterable[dict] = system_output_object
        self.gen_kwargs = gen_kwargs
        self._data: Table = feature_table
        # _samples_over_bucket_true: Dict(feature_name, bucket_name, sample_id_true_label):
        # samples in different buckets
        self._samples_over_bucket_true = {}
        self._samples_over_bucket_pred = {}
        # _performances_over_bucket: performance in different bucket: Dict(feature_name, bucket_name, performance)
        self._performances_over_bucket = {}

        self._path_pre_computed_models = None
        self.dict_pre_computed_models = None

        scriptpath = os.path.dirname(__file__)
        if self._info.dataset_name and self._info.task_name:
            self._path_pre_computed_models = os.path.join(scriptpath, "../pre_computed/" + self._info.task_name.replace("-","_") + "/" + self._info.dataset_name+"/")
            # print(self._path_pre_computed_models)
            # print(os.path.isdir(self._path_pre_computed_models))
            # exit()
            if os.path.isdir(self._path_pre_computed_models):
                self.dict_pre_computed_models = self.get_pre_computed_features()




    def get_pre_computed_features(self):
        dict_pre_computed_models = {}
        for feature_name in self._info.features.get_pre_computed_features():
            if os.path.exists(self._path_pre_computed_models):
                # print('load the hard dictionary of entity span in test set...')
                # print(self._path_pre_computed_models + "/" + feature_name + ".pkl")
                fread = open(self._path_pre_computed_models + "/" + feature_name + ".pkl", 'rb')
                dict_pre_computed_models[feature_name] = pickle.load(fread)
            else:
                raise ValueError("can not load hard dictionary" + feature_name + "\t" + self._path_pre_computed_models)

        return dict_pre_computed_models




    @staticmethod
    def get_bucket_feature_value(feature_name:str):
        return "self._get_" + feature_name

    # define function for incomplete features
    def _get_sentence_length(self, existing_features: dict):
        return len(existing_features["sentence"].split(" "))




    def _get_eCon_value(self, span_dic:dict, span_text:str, span_tag:str):
        """
        Since keys and values of span_dic have been lower-cased, we also need to lowercase span_tag and span_text

        """
        span_tag = span_tag.lower()
        span_text = span_text.lower()

        eCon_value = 0.0
        if span_text in span_dic.keys():
            if span_tag in span_dic[span_text]:
                eCon_value = float(span_dic[span_text][span_tag])
        return eCon_value

    def _get_eFre_value(self, span_dic, span_text, span_tag):
        eFre_value = 0.0
        span_tag = span_tag.lower()
        span_text = span_text.lower()
        if span_text in span_dic.keys():
            eFre_value = float(span_dic[span_text])
        return eFre_value


    def _complete_feature_raw_span_features(self, sentence, tags):
        # span_text, span_len, span_pos, span_tag
        chunks = get_chunks(tags)
        span_dics = []
        span_dic = {}
        for chunk in chunks:
            tag, sid, eid = chunk
            #span_text = ' '.join(sentence[sid:eid]).lower()
            span_text = ' '.join(sentence[sid:eid])
            span_len = eid - sid
            span_pos = (sid, eid)
            span_dic = {'span_text': span_text, 'span_len': span_len, 'span_pos': span_pos, 'span_tag': tag}
            # print('span_dic: ',span_dic)
            span_dics.append(span_dic)
        # self.span_dics = span_dics
        return span_dics

    def _complete_feature_advanced_span_features(self, sentence, tags):
        span_dics = self._complete_feature_raw_span_features(sentence, tags)



        eCon_dic = self.dict_pre_computed_models['eCon'] if self.dict_pre_computed_models else None
        eFre_dic = self.dict_pre_computed_models['eFre']if self.dict_pre_computed_models else None

        span_dics_list= []
        for span_dic in span_dics:
            span_text = span_dic['span_text']
            span_tag = span_dic['span_tag']
            span_pos = span_dic['span_pos']

            if self.dict_pre_computed_models:
                # compute the entity-level label consistency...
                eCon_value = self._get_eCon_value(eCon_dic, span_text, span_tag)
                if 'eCon' not in span_dic:
                    span_dic['eCon'] = eCon_value
            else:
                span_dic['eCon'] = 0
            if self.dict_pre_computed_models:
                # compute the entity-level frequency...
                eFre_value = self._get_eFre_value(eFre_dic, span_text, span_tag)
                if 'eFre' not in span_dic:
                    span_dic['eFre'] = eFre_value
            else:
                span_dic['eFre'] = 0


            # print('span_dic list: ', span_dic)
            span_dics_list.append(span_dic)
        return span_dics_list



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
            tokens = dict_sysout["tokens"]
            true_tags = dict_sysout["true_tags"]
            pred_tags = dict_sysout["pred_tags"]

            dict_sysout["sentence_length"] = len(tokens)

            dict_sysout["true_entity_info"] = self._complete_feature_advanced_span_features(tokens, true_tags)
            dict_sysout["pred_entity_info"] = self._complete_feature_advanced_span_features(tokens, pred_tags)

            # for bucket_feature in bucket_features:
            #     feature_value = eval(NERExplainaboardBuilder.get_bucket_feature_value(bucket_feature))(dict_sysout)
            #     dict_sysout[bucket_feature] = feature_value
            if self._data == None:
                self._data = {}
            self._data[_id] = dict_sysout
            yield _id, dict_sysout


    def get_overall_performance(self):
        predicted_labels,true_labels = [], []

        true_tags_list = []
        pred_tags_list = []

        for _id, feature_table in self._data.items():

            true_tags_list.append(feature_table["true_tags"])
            pred_tags_list.append(feature_table["pred_tags"])

        for metric_name in self._info.metric_names:

            res_json = eval(metric_name)(true_tags_list, pred_tags_list)

            overall_value = res_json["f1"]
            # overall_value = f1_score_seqeval(true_tags_list, pred_tags_list)["f1"]

            # metric_name = "F1score_seqeval"
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
        feature_to_sample_address_to_value_true = {}
        feature_to_sample_address_to_value_pred = {}



        # Preparation for bucketing
        for _id, feature_table in sysout_iterator:

            # true tag
            true_entity_info_list = feature_table["true_entity_info"]
            for span_info in true_entity_info_list:
                span_text = span_info["span_text"]
                span_pos  = span_info["span_pos"]
                span_label = span_info["span_tag"]


                span_address = str(_id) + "|||" + str(span_pos[0]) + "|||" + str(span_pos[1]) + "|||" + span_text + "|||" + span_label


                for feature_name in self._info.features.get_bucket_features():
                    if feature_name not in feature_to_sample_address_to_value_true.keys():
                        feature_to_sample_address_to_value_true[feature_name] = {}
                    else:
                        if feature_name not in feature_table.keys():
                            feature_to_sample_address_to_value_true[feature_name][span_address] = span_info[feature_name]
                        else:
                            feature_to_sample_address_to_value_true[feature_name][span_address] = feature_table[feature_name]

            # pred tag
            pred_entity_info_list = feature_table["pred_entity_info"]
            for span_info in pred_entity_info_list:
                span_text = span_info["span_text"]
                span_pos  = span_info["span_pos"]
                span_label = span_info["span_tag"]


                span_address = str(_id) + "|||" + str(span_pos[0]) + "|||" + str(span_pos[1]) + "|||" + span_text + "|||" + span_label


                for feature_name in self._info.features.get_bucket_features():
                    if feature_name not in feature_to_sample_address_to_value_pred.keys():
                        feature_to_sample_address_to_value_pred[feature_name] = {}
                    else:
                        if feature_name not in feature_table.keys():
                            feature_to_sample_address_to_value_pred[feature_name][span_address] = span_info[feature_name]
                        else:
                            feature_to_sample_address_to_value_pred[feature_name][span_address] = feature_table[feature_name]




        # Bucketing
        for feature_name in tqdm(self._info.features.get_bucket_features(), desc="bucketing"):

            _bucket_info = ""
            if feature_name in self._info.features.keys():
                _bucket_info = self._info.features[feature_name].bucket_info
            else:
                # print(self._info.features)
                _bucket_info = self._info.features["true_entity_info"].feature.feature[feature_name].bucket_info

            # print(f"Feature Name: {feature_name}\n"
            #       f"Bucket Hyper:\n function_name: {_bucket_info._method} \n"
            #       f"bucket_number: {_bucket_info._number}\n"
            #       f"bucket_setting: {_bucket_info._setting}\n")

            self._samples_over_bucket_true[feature_name] = eval(_bucket_info._method)(
                                dict_obj = feature_to_sample_address_to_value_true[feature_name],
                                bucket_number = _bucket_info._number,
                                bucket_setting = _bucket_info._setting)



            # print(f"debug-1: {self._samples_over_bucket_true[feature_name]}")
            self._samples_over_bucket_pred[feature_name] = bucket_attribute_specified_bucket_interval(
                                dict_obj = feature_to_sample_address_to_value_pred[feature_name],
                                bucket_number = _bucket_info._number,
                                bucket_setting = self._samples_over_bucket_true[feature_name].keys())


            # print(f"self._samples_over_bucket.keys():\n{self._samples_over_bucket_true.keys()}")

            # evaluating bucket: get bucket performance
            self._performances_over_bucket[feature_name] = self.get_bucket_performance(feature_name)


    """
    Get bucket samples (with mis-predicted entities) for each bucket given a feature (e.g., length)
    """
    def get_bucket_cases_ner(self, feature_name:str, bucket_interval) -> list:
        # predict:  2_3 -> NER
        dict_pos2tag_pred = {}
        for k_bucket_eval, spans_pred in self._samples_over_bucket_pred[feature_name].items():
            if k_bucket_eval != bucket_interval:
                continue
            for span_pred in spans_pred:
                pos_pred = "|||".join(span_pred.split("|||")[0:4])
                tag_pred = span_pred.split("|||")[-1]
                dict_pos2tag_pred[pos_pred] = tag_pred
        # print(dict_pos2tag_pred)

        # true:  2_3 -> NER
        dict_pos2tag = {}
        for k_bucket_eval, spans in self._samples_over_bucket_true[feature_name].items():
            if k_bucket_eval != bucket_interval:
                continue
            for span in spans:
                pos = "|||".join(span.split("|||")[0:4])
                tag = span.split("|||")[-1]
                dict_pos2tag[pos] = tag

        errorCase_list = []
        for pos, tag in dict_pos2tag.items():

            true_label = tag
            pred_label = ""
            sent_id = int(pos.split("|||")[0])
            span = pos.split("|||")[-1]
            system_output_id = self._data[int(sent_id)]["id"]


            span_sentence = " ".join(self._data[sent_id]["tokens"])

            if pos in dict_pos2tag_pred.keys():
                pred_label = dict_pos2tag_pred[pos]
                if true_label == pred_label:
                    continue
            else:
                pred_label = "O"
            #error_case = span+ "|||" + span_sentence + "|||" + true_label + "|||" + pred_label
            error_case = {
                "span":span,
                "text":str(system_output_id),
                "true_label":true_label,
                "predicted_label":pred_label,
            }
            errorCase_list.append(error_case)

        for pos, tag in dict_pos2tag_pred.items():

            true_label = ""
            pred_label = tag

            sent_id = int(pos.split("|||")[0])
            span = pos.split("|||")[-1]
            span_sentence = " ".join(self._data[sent_id]["tokens"])
            system_output_id = self._data[int(sent_id)]["id"]
            # print(span_sentence)

            if pos in dict_pos2tag.keys():
                true_label = dict_pos2tag[pos]
                if true_label == pred_label:
                    continue
            else:
                true_label = "O"
            #error_case = span + "|||" + span_sentence + "|||" + true_label + "|||" + pred_label
            error_case = {
                "span":span,
                "text":system_output_id,
                "true_label":true_label,
                "predicted_label":pred_label,
            }
            errorCase_list.append(error_case)

        return errorCase_list

    def get_bucket_performance(self, feature_name:str):
        """
        This function defines how to get bucket-level performance w.r.t a given feature (e.g., sentence length)
        :param feature_name: the name of a feature, e.g., sentence length
        :return: bucket_name_to_performance: a dictionary that maps bucket names to bucket performance
        """

        bucket_name_to_performance = {}
        for bucket_interval, spans_true in self._samples_over_bucket_true[feature_name].items():

            spans_pred = []
            if bucket_interval not in self._samples_over_bucket_pred[feature_name].keys():
                raise ValueError("Predict Label Bucketing Errors")
            else:
                spans_pred = self._samples_over_bucket_pred[feature_name][bucket_interval]

            """
            Get bucket samples for ner task
            """
            bucket_samples = self.get_bucket_cases_ner(feature_name, bucket_interval)

            for metric_name in self._info.metric_names:
                """
                # Note that: for NER task, the bucket-wise evaluation function is a little different from overall evaluation function
                # for overall: f1_score_seqeval
                # for bucket:  f1_score_seqeval_bucket_bucket                
                """
                f1, p, r = eval(metric_name+"_bucket")(spans_pred, spans_true)


                bucket_name_to_performance[bucket_interval] = []
                bucket_performance = BucketPerformance(bucket_name=bucket_interval,
                                                       metric_name=metric_name,
                                                       value=format(f1, '.4g'),
                                                       confidence_score_low= 0.0,
                                                       confidence_score_up= 0.0,
                                                       n_samples=len(spans_pred),
                                                       bucket_samples=bucket_samples)

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


