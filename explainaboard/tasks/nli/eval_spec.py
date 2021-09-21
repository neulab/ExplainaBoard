# -*- coding: utf-8 -*-
import explainaboard.error_analysis as ea
import explainaboard.data_utils as du
import os
import numpy
from collections import OrderedDict


def get_aspect_value(sent1_list, sent2_list, sample_list_tag, sample_list_tag_pred, dict_aspect_func):
    dict_span2aspect_val = {}
    dict_span2aspect_val_pred = {}

    for aspect, fun in dict_aspect_func.items():
        dict_span2aspect_val[aspect] = {}
        dict_span2aspect_val_pred[aspect] = {}

    # for error analysis
    dict_sid2sentpair = {}

    sample_id = 0
    for sent1, sent2, tag, tag_pred in zip(sent1_list, sent2_list, sample_list_tag, sample_list_tag_pred):

        word_list1 = ea.word_segment(sent1).split(" ")
        word_list2 = ea.word_segment(sent2).split(" ")

        # for saving errorlist -- fine-grained version
        dict_sid2sentpair[str(sample_id)] = ea.format4json2(
            ea.format4json2(sent1) + "|||" + ea.format4json2(sent2))

        sent1_length = len(word_list1)
        sent2_length = len(word_list2)

        sent_pos = ea.tuple2str((sample_id, tag))
        sent_pos_pred = ea.tuple2str((sample_id, tag_pred))

        hypo = [ea.word_segment(sent2)]
        refs = [[ea.word_segment(sent1)]]

        # bleu = sacrebleu.corpus_bleu(hypo, refs).score * 0.01

        # aspect = "bleu"
        # if aspect in dict_aspect_func.keys():
        # 	dict_span2aspect_val["bleu"][sent_pos] = bleu
        # 	dict_span2aspect_val_pred["bleu"][sent_pos_pred] = bleu

        # Sentence Length: sentALen
        aspect = "sentALen"
        if aspect in dict_aspect_func.keys():
            dict_span2aspect_val[aspect][sent_pos] = float(sent1_length)
            dict_span2aspect_val_pred[aspect][sent_pos_pred] = float(sent1_length)

        # Sentence Length: sentBLen
        aspect = "sentBLen"
        if aspect in dict_aspect_func.keys():
            dict_span2aspect_val["sentBLen"][sent_pos] = float(sent2_length)
            dict_span2aspect_val_pred[aspect][sent_pos_pred] = float(sent2_length)

        # The difference of sentence length: senDeltaLen
        aspect = "A-B"
        if aspect in dict_aspect_func.keys():
            dict_span2aspect_val["A-B"][sent_pos] = float(sent1_length - sent2_length)
            dict_span2aspect_val_pred[aspect][sent_pos_pred] = float(sent1_length - sent2_length)

        # "A+B"
        aspect = "A+B"
        if aspect in dict_aspect_func.keys():
            dict_span2aspect_val["A+B"][sent_pos] = float(sent1_length + sent2_length)
            dict_span2aspect_val_pred[aspect][sent_pos_pred] = float(sent1_length + sent2_length)

        # "A/B"
        aspect = "A/B"
        if aspect in dict_aspect_func.keys():
            dict_span2aspect_val["A/B"][sent_pos] = float(sent1_length * 1.0 / sent2_length)
            dict_span2aspect_val_pred[aspect][sent_pos_pred] = float(sent1_length * 1.0 / sent2_length)

        # Tag: tag
        aspect = "tag"
        if aspect in dict_aspect_func.keys():
            dict_span2aspect_val["tag"][sent_pos] = tag
            dict_span2aspect_val_pred[aspect][sent_pos_pred] = tag

        sample_id += 1
    # print(dict_span2aspect_val["bleu"])
    return dict_span2aspect_val, dict_span2aspect_val_pred, dict_sid2sentpair


def evaluate(task_type="ner", analysis_type="single", systems=[], dataset_name = 'dataset_name', model_name = 'model_name', output_filename="./output.json", is_print_ci=False,
             is_print_case=False, is_print_ece=False):

    path_text = systems[0] if analysis_type == "single" else ""
    path_comb_output = "model_name" + "/" + path_text.split("/")[-1]
    dict_aspect_func, dict_precomputed_path, obj_json = ea.load_task_conf(task_dir=os.path.dirname(__file__))

    sent1_list, sent2_list, true_label_list, pred_label_list = du.tsv_to_lists(path_text, col_ids=(0,1,2,3))

    error_case_list = []
    if is_print_case:
        error_case_list = ea.get_error_case_classification(true_label_list, pred_label_list, sent1_list, sent2_list)
        print(" -*-*-*- the number of error casse:\t", len(error_case_list))

    # Confidence Interval of Holistic Performance
    confidence_low, confidence_up = 0, 0
    if is_print_ci:
        confidence_low, confidence_up = ea.compute_confidence_interval_acc(true_label_list, pred_label_list,
                                                                           n_times=100)

    dict_span2aspect_val, dict_span2aspect_val_pred, dict_sid2sentpair = get_aspect_value(sent1_list, sent2_list,
                                                                                      true_label_list, pred_label_list,
                                                                                      dict_aspect_func)

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
        dict_bucket2f1[aspect] = get_bucket_acc_with_error_case(dict_bucket2span[aspect],
                                                                dict_bucket2span_pred[aspect], dict_sid2sentpair,
                                                                is_print_ci, is_print_case)
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


    dict_fine_grained = {}
    for aspect, metadata in dict_bucket2f1.items():
        dict_fine_grained[aspect] = []
        for bucket_name, v in metadata.items():
            # print("---------debug--bucket name old---")
            # print(bucket_name)
            bucket_name = ea.beautify_interval(bucket_name)
            # print("---------debug--bucket name new---")
            # print(bucket_name)

            # bucket_value = format(v[0]*100,'.4g')
            bucket_value = format(v[0], '.4g')
            n_sample = v[1]
            confidence_low = format(v[2], '.4g')
            confidence_up = format(v[3], '.4g')

            # for saving errorlist -- fine_grained version
            bucket_error_case = v[4]

            # instantiation
            dict_fine_grained[aspect].append({"bucket_name": bucket_name, "bucket_value": bucket_value, "num": n_sample,
                                             "confidence_low": confidence_low, "confidence_up": confidence_up,
                                             "bucket_error_case": bucket_error_case})

    obj_json["task"] = task_type
    obj_json["data"]["language"] = "English"
    obj_json["data"]["bias"] = dict_aspect2bias
    obj_json["data"]["name"] = dataset_name

    obj_json["model"]["name"] = model_name
    obj_json["model"]["results"]["overall"]["performance"] = holistic_performance
    obj_json["model"]["results"]["overall"]["confidence_low"] = confidence_low
    obj_json["model"]["results"]["overall"]["confidence_up"] = confidence_up
    obj_json["model"]["results"]["fine_grained"] = dict_fine_grained

    # add errorAnalysis -- holistic
    obj_json["model"]["results"]["overall"]["error_case"] = error_case_list

    # for Calibration
    ece = 0
    dic_calibration = None
    if is_print_ece:
        ece, dic_calibration = ea.calculate_ece_by_file(path_text, prob_col=4, answer_cols=(2,3),
                                                        size_of_bin=10, dataset="dataset_name", model="model_name")

    obj_json["model"]["results"]["calibration"] = dic_calibration

    ea.save_json(obj_json, output_filename)


def get_bucket_acc_with_error_case(dict_bucket2span, dict_bucket2span_pred, dict_sid2sentpair, is_print_ci,
                                   is_print_case):
    # The structure of span_true or span_pred
    # 2345|||Positive
    # 2345 represents sentence id
    # Positive represents the "label" of this instance

    dict_bucket2f1 = {}

    for bucket_interval, spans_true in dict_bucket2span.items():
        spans_pred = []
        if bucket_interval not in dict_bucket2span_pred.keys():
            raise ValueError("Predict Label Bucketing Errors")
        else:
            spans_pred = dict_bucket2span_pred[bucket_interval]

        # loop over samples from a given bucket
        error_case_bucket_list = []
        if is_print_case:
            for info_true, info_pred in zip(spans_true, spans_pred):
                sid_true, label_true = info_true.split("|||")
                sid_pred, label_pred = info_pred.split("|||")
                if sid_true != sid_pred:
                    continue

                sent = dict_sid2sentpair[sid_true]
                if label_true != label_pred:
                    error_case_info = label_true + "|||" + label_pred + "|||" + sent
                    error_case_bucket_list.append(error_case_info)

        accuracy_each_bucket = ea.accuracy(spans_pred, spans_true)
        confidence_low, confidence_up = 0, 0
        if is_print_ci:
            confidence_low, confidence_up = ea.compute_confidence_interval_acc(spans_pred, spans_true)
        dict_bucket2f1[bucket_interval] = [accuracy_each_bucket, len(spans_true), confidence_low, confidence_up,
                                           error_case_bucket_list]

    return ea.sort_dict(dict_bucket2f1)


