# -*- coding: utf-8 -*-
from collections import OrderedDict

import explainaboard.error_analysis as ea
import explainaboard.data_utils as du
import numpy
import os


def get_aspect_value(sent_list, sample_list_tag, sample_list_tag_pred, dict_aspect_func):
    dict_span2aspect_val = {}
    dict_span2aspect_val_pred = {}

    for aspect, fun in dict_aspect_func.items():
        dict_span2aspect_val[aspect] = {}
        dict_span2aspect_val_pred[aspect] = {}

    # maintain it for print error case
    dict_sid2sent = {}

    sample_id = 0
    for sent, tag, tag_pred in zip(sent_list, sample_list_tag, sample_list_tag_pred):

        dict_sid2sent[str(sample_id)] = ea.format4json2(sent)

        word_list = ea.word_segment(sent).split(" ")

        sent_length = len(word_list)

        sent_pos = ea.tuple2str((sample_id, tag))
        sent_pos_pred = ea.tuple2str((sample_id, tag_pred))

        # Sentence Length: sentALen
        aspect = "sLen"
        if aspect in dict_aspect_func.keys():
            dict_span2aspect_val[aspect][sent_pos] = float(sent_length)
            dict_span2aspect_val_pred[aspect][sent_pos_pred] = float(sent_length)

        # Tag: tag
        aspect = "tag"  ############## MUST Be Gold Tag for text classification task
        if aspect in dict_aspect_func.keys():
            dict_span2aspect_val[aspect][sent_pos] = tag
            dict_span2aspect_val_pred[aspect][sent_pos_pred] = tag

        sample_id += 1

    # print(dict_span2aspect_val["bleu"])
    return dict_span2aspect_val, dict_span2aspect_val_pred, dict_sid2sent


def process_all(file_path, size_of_bin=10, dataset='atis', model='lstm-self-attention'):
    """

    :param file_path: the file_path is the path to your file.

    And the path must include file name.

    the file name is in this format: test_dataset_model.tsv.

    the file_path must in the format: /root/path/to/your/file/test_dataset.tsv

    The file must in this format:
    sentence\tground_truth\tpredict_label\tprobability\tright_or_not
    if prediction is right, right_or_not is assigned to 1, otherwise 0.

    :param size_of_bin: the numbers of how many bins

    :param dataset: the name of the dataset

    :param model: the name of the model

    :return:
    ece :the ece of this file
    dic :the details of the ECE information in json format
    """

    # import json

    probability_list, right_or_not_list = du.get_probability_right_or_not(file_path, prob_col=3, right_or_not_col=4)

    raw_list = list(zip(probability_list, right_or_not_list))

    bin_list = ea.divide_into_bin(size_of_bin, raw_list)

    ece = ea.calculate_ece(bin_list)
    dic = OrderedDict()
    dic['dataset-name'] = dataset
    dic['model-name'] = model
    dic['ECE'] = ece
    dic['details'] = []
    basic_width = 1 / size_of_bin
    for i in range(len(bin_list)):
        tem_dic = {}

        bin_name = format(i * basic_width, '.2g') + '-' + format((i + 1) * basic_width, '.2g')
        tem_dic = {'interval': bin_name, 'average_accuracy': bin_list[i][1], 'average_confidence': bin_list[i][0],
                   'samples_number_in_this_bin': bin_list[i][2]}
        dic['details'].append(tem_dic)

    return ece, dic

def get_error_case(sent_list, true_label_list, pred_label_list):
    error_case_list = []
    for sent, true_label, pred_label in zip(sent_list, true_label_list, pred_label_list):
        if true_label != pred_label:
            error_case_list.append(true_label + "|||" + pred_label + "|||" + ea.format4json2(sent))
    return error_case_list

def evaluate(task_type="ner", analysis_type="single", systems=[], dataset_name = 'dataset_name', model_name = 'model_name', output_filename="./output.json", is_print_ci=False,
             is_print_case=False, is_print_ece=False):

    path_text = systems[0] if analysis_type == "single" else ""
    path_comb_output = "model_name" + "/" + path_text.split("/")[-1]
    dict_aspect_func, dict_precomputed_path, obj_json = ea.load_task_conf(task_dir=os.path.dirname(__file__))

    sent_list, true_label_list, pred_label_list = file_to_list_tc(path_text)

    error_case_list = []
    if is_print_case:
        error_case_list = get_error_case(sent_list, true_label_list, pred_label_list)
        print(" -*-*-*- the number of error casse:\t", len(error_case_list))

    # Confidence Interval of Holistic Performance
    confidence_low, confidence_up = 0, 0
    if is_print_ci:
        confidence_low, confidence_up = ea.compute_confidence_interval_acc(true_label_list, pred_label_list,
                                                                           n_times=1000)

    dict_span2aspect_val, dict_span2aspect_val_pred, dict_sid2sent = get_aspect_value(sent_list, true_label_list,
                                                                                  pred_label_list, dict_aspect_func)

    holistic_performance = ea.accuracy(true_label_list, pred_label_list)
    holistic_performance = format(holistic_performance, '.3g')

    print("------------------ Holistic Result----------------------")
    print(holistic_performance)

    # print(f1(list_true_tags_token, list_pred_tags_token)["f1"])

    dict_bucket2span = {}
    dict_bucket2span_pred = {}
    dict_bucket2f1 = {}
    aspect_names = []

    for aspect, func in dict_aspect_func.items():
        # print(aspect, dict_span2aspect_val[aspect])
        dict_bucket2span[aspect] = ea.select_bucketing_func(func[0], func[1], dict_span2aspect_val[aspect])
        # print(aspect, dict_bucket2span[aspect])
        # exit()
        dict_bucket2span_pred[aspect] = ea.bucket_attribute_specified_bucket_interval(dict_span2aspect_val_pred[aspect],
                                                                                      dict_bucket2span[aspect].keys())
        # dict_bucket2span_pred[aspect] = __select_bucketing_func(func[0], func[1], dict_span2aspect_val_pred[aspect])
        dict_bucket2f1[aspect] = ea.get_bucket_acc_with_error_case(dict_bucket2span[aspect], dict_bucket2span_pred[aspect],
                                                                dict_sid2sent, is_print_ci, is_print_case)
        aspect_names.append(aspect)
    print("aspect_names: ", aspect_names)

    print("------------------ Breakdown Performance")
    for aspect in dict_aspect_func.keys():
        ea.print_dict(dict_bucket2f1[aspect], aspect)
    print("")

    # Calculate databias w.r.t numeric attributes
    dict_aspect2bias = {}
    for aspect, aspect2Val in dict_span2aspect_val.items():
        if type(list(aspect2Val.values())[0]) != type("string"):
            dict_aspect2bias[aspect] = numpy.average(list(aspect2Val.values()))

    print("------------------ Dataset Bias")
    for k, v in dict_aspect2bias.items():
        print(k + ":\t" + str(v))
    print("")

    def beautify_interval(interval):

        if type(interval[0]) == type("string"):  ### pay attention to it
            return interval[0]
        else:
            if len(interval) == 1:
                bk_name = '(' + format(float(interval[0]), '.3g') + ',)'
                return bk_name
            else:
                range1_r = '(' + format(float(interval[0]), '.3g') + ','
                range1_l = format(float(interval[1]), '.3g') + ')'
                bk_name = range1_r + range1_l
                return bk_name

    dict_fine_grained = {}
    for aspect, metadata in dict_bucket2f1.items():
        dict_fine_grained[aspect] = []
        for bucket_name, v in metadata.items():
            # print("---------debug--bucket name old---")
            # print(bucket_name)
            bucket_name = beautify_interval(bucket_name)
            # print("---------debug--bucket name new---")
            # print(bucket_name)

            # bucket_value = format(v[0]*100,'.4g')
            bucket_value = format(v[0], '.4g')
            n_sample = v[1]
            confidence_low = format(v[2], '.4g')
            confidence_up = format(v[3], '.4g')
            bucket_error_case = v[4]

            # instantiation
            dict_fine_grained[aspect].append({"bucket_name": bucket_name, "bucket_value": bucket_value, "num": n_sample,
                                             "confidence_low": confidence_low, "confidence_up": confidence_up,
                                             "bucket_error_case": bucket_error_case})

    obj_json["task"] = task_type
    obj_json["data"]["language"] = "English"
    obj_json["data"]["bias"] = dict_aspect2bias
    obj_json["data"]["output"] = path_comb_output
    obj_json["data"]["name"] = dataset_name
    obj_json["model"]["name"] = model_name

    obj_json["model"]["results"]["overall"]["error_case"] = error_case_list
    obj_json["model"]["results"]["overall"]["performance"] = holistic_performance
    obj_json["model"]["results"]["overall"]["confidence_low"] = confidence_low
    obj_json["model"]["results"]["overall"]["confidence_up"] = confidence_up
    obj_json["model"]["results"]["fine_grained"] = dict_fine_grained

    ece = 0
    dic_calibration = None
    if is_print_ece:
        ece, dic_calibration = process_all(path_text,
                                           size_of_bin=10, dataset="dataset_name", model="model_name")

    obj_json["model"]["results"]["calibration"] = dic_calibration
    # print(dic_calibration)

    ea.save_json(obj_json, "./instantiate.json")
    ea.save_json(obj_json, output_filename)


def file_to_list_tc(path_file):
    sent_list = []
    true_label_list = []
    pred_label_list = []
    fin = open(path_file, "r")
    for line in fin:
        line = line.rstrip("\n")
        if len(line.split("\t")) != 5:
            continue
        sent, true_label, pred_label = line.split("\t")[0], line.split("\t")[1], line.split("\t")[2]
        sent_list.append(sent)
        true_label_list.append(true_label)
        pred_label_list.append(pred_label)

    fin.close()
    return sent_list, true_label_list, pred_label_list