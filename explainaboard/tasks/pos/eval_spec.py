# -*- coding: utf-8 -*-
from random import choices

import explainaboard.error_analysis as ea
import pickle
import numpy
import os


def get_aspect_value(test_word_sequences, test_true_tag_sequences, test_word_sequences_sent,
                   test_true_tag_sequences_sent, dict_precomputed_path, dict_aspect_func):
    dict_precomputed_model = {}
    for aspect, path in dict_precomputed_path.items():
        print("path:\t" + path)
        if ea.os.path.exists(path):
            print('load the hard dictionary of entity span in test set...')
            fread = open(path, 'rb')
            dict_precomputed_model[aspect] = pickle.load(fread)
        else:
            raise ValueError("can not load hard dictionary" + aspect + "\t" + path)

    dict_span2aspect_val = {}
    for aspect, fun in dict_aspect_func.items():
        dict_span2aspect_val[aspect] = {}

    dict_pos2sid = ea.get_pos2sentid(test_word_sequences_sent)
    dict_ap2rp = ea.get_token_position(test_word_sequences_sent)

    dict_span2sid = {}
    dict_chunkid2span = {}
    for token_id, token in enumerate(test_word_sequences):

        token_type = test_true_tag_sequences[token_id]
        token_pos = str(token_id) + "_" + str(token) + "_" + token_type
        token_sentid = dict_pos2sid[token_id]
        sLen = float(len(test_word_sequences_sent[token_sentid]))

        dict_span2sid[token_pos] = token_sentid
        dict_chunkid2span[token_pos] = token + "|||" + ea.format4json(' '.join(test_word_sequences_sent[token_sentid]))

        # Sentence Length: sentLen
        aspect = "sLen"
        if aspect in dict_aspect_func.keys():
            dict_span2aspect_val[aspect][token_pos] = sLen

        # Sentence Length: tokLen
        aspect = "tLen"
        if aspect in dict_aspect_func.keys():
            dict_span2aspect_val[aspect][token_pos] = float(len(token))
            # if float(len(token)) == 1:
            #     print(token)

        # Relative Position: relPos
        aspect = "rPos"
        if aspect in dict_aspect_func.keys():
            dict_span2aspect_val[aspect][token_pos] = (dict_ap2rp[token_id]) * 1.0 / sLen

        # Tag: tag
        aspect = "tag"
        if aspect in dict_aspect_func.keys():
            dict_span2aspect_val[aspect][token_pos] = token_type

    return dict_span2aspect_val, dict_span2sid, dict_chunkid2span


# def tuple2str(triplet):
#     res = ""
#     for v in triplet:
#         res += str(v) + "_"
#     return res.rstrip("_")


def evaluate(task_type="ner", analysis_type="single", systems=[], output_filename="./output.json", is_print_ci=False,
             is_print_case=False, is_print_ece=False):

    path_text = systems[0] if analysis_type == "single" else ""
    path_comb_output = "model_name" + "/" + path_text.split("/")[-1]
    dict_aspect_func, dict_precomputed_path, obj_json = ea.load_task_conf(task_dir=os.path.dirname(__file__))

    list_text_sent, list_text_token = ea.read_single_column(path_text, 0)
    list_true_tags_sent, list_true_tags_token = ea.read_single_column(path_text, 1)
    list_pred_tags_sent, list_pred_tags_token = ea.read_single_column(path_text, 2)

    dict_span2aspect_val, dict_span2sid, dict_chunkid2span = get_aspect_value(list_text_token, list_true_tags_token,
                                                                           list_text_sent,
                                                                           list_true_tags_sent, dict_precomputed_path,
                                                                           dict_aspect_func)
    dict_span2aspect_val_pred, dict_span2sid_pred, dict_chunkid2span_pred = get_aspect_value(list_text_token,
                                                                                          list_pred_tags_token,
                                                                                          list_text_sent,
                                                                                          list_pred_tags_sent,
                                                                                          dict_precomputed_path,
                                                                                          dict_aspect_func)

    print(len(dict_chunkid2span), len(dict_chunkid2span_pred))

    holistic_performance = ea.accuracy(list_true_tags_token, list_pred_tags_token)

    confidence_low_overall, confidence_up_overall = 0, 0
    if is_print_ci:
        confidence_low_overall, confidence_up_overall = ea.compute_confidence_interval_f1(dict_span2sid.keys(),
                                                                                          dict_span2sid_pred.keys(),
                                                                                          dict_span2sid,
                                                                                          dict_span2sid_pred,
                                                                                          n_times=1000)

    print("------------------ Holistic Result")
    print()
    print(holistic_performance)

    print("confidence_low_overall:\t", confidence_low_overall)
    print("confidence_up_overall:\t", confidence_up_overall)

    dict_bucket2span = {}
    dict_bucket2span_pred = {}
    dict_bucket2f1 = {}
    aspect_names = []
    error_case_list = []

    for aspect, func in dict_aspect_func.items():
        # print(aspect, dict_span2aspect_val[aspect])
        dict_bucket2span[aspect] = ea.select_bucketing_func(func[0], func[1], dict_span2aspect_val[aspect])
        # print(aspect, dict_bucket2span[aspect])
        # exit()
        dict_bucket2span_pred[aspect] = ea.bucket_attribute_specified_bucket_interval(dict_span2aspect_val_pred[aspect],
                                                                                      dict_bucket2span[aspect].keys())
        dict_bucket2f1[aspect], error_case_list = get_bucket_f1(dict_bucket2span[aspect],
                                                                dict_bucket2span_pred[aspect], dict_span2sid,
                                                                dict_span2sid_pred, dict_chunkid2span,
                                                                dict_chunkid2span_pred, is_print_ci, is_print_case)
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
            bucket_value = format(float(v[0]) * 100, '.4g')
            n_sample = v[1]
            confidence_low = format(float(v[2]) * 100, '.4g')
            confidence_up = format(float(v[3]) * 100, '.4g')
            error_entity_list = v[4]

            # instantiation
            dict_fine_grained[aspect].append({"bucket_name": bucket_name, "bucket_value": bucket_value, "num": n_sample,
                                             "confidence_low": confidence_low, "confidence_up": confidence_up,
                                             "bucket_error_case": error_entity_list})

    obj_json["task"] = task_type
    obj_json["data"]["language"] = "English"
    obj_json["data"]["bias"] = dict_aspect2bias
    obj_json["data"]["output"] = path_comb_output
    obj_json["model"]["results"]["overall"]["performance"] = holistic_performance
    obj_json["model"]["results"]["overall"]["confidence_low"] = confidence_low_overall
    obj_json["model"]["results"]["overall"]["confidence_up"] = confidence_up_overall
    obj_json["model"]["results"]["fine_grained"] = dict_fine_grained

    obj_json["model"]["results"]["overall"]["error_case"] = error_case_list

    ea.save_json(obj_json, output_filename)


def get_error_case(dict_pos2tag, dict_pos2tag_pred, dict_chunkid2span_sent, dict_chunkid2span_sent_pred):
    # print("debug-1:")
    # print()

    error_case_list = []
    for pos, tag in dict_pos2tag.items():

        true_label = tag
        pred_label = ""
        # print(dict_chunkid2span_sent.keys())
        if pos + "_" + tag not in dict_chunkid2span_sent.keys():
            continue
        span_sentence = dict_chunkid2span_sent[pos + "_" + tag]

        if pos in dict_pos2tag_pred.keys():
            pred_label = dict_pos2tag_pred[pos]
            if true_label == pred_label:
                continue
        else:
            # pred_label = "O"
            continue

        error_case = ea.format4json2(span_sentence) + "|||" + true_label + "|||" + pred_label

        # if pred_label == "O":
        # 	print(error_case)
        # 	print(len(dict_pos2tag), len(dict_pos2tag_pred))
        # 	print(pos)

        error_case_list.append(error_case)

    # print(error_case_list)
    return error_case_list


def compute_confidence_interval_f1(spans_true, spans_pred, dict_span2sid, dict_span2sid_pred, n_times=100):
    n_data = len(dict_span2sid)
    sample_rate = ea.get_sample_rate(n_data)
    n_sampling = int(n_data * sample_rate)
    print("sample_rate:\t", sample_rate)
    print("n_sampling:\t", n_sampling)

    dict_sid2span_salient = {}
    for span in spans_true:
        # print(span)
        if len(span.split("_")) != 3:
            break
        sid = dict_span2sid[span]
        if sid in dict_sid2span_salient.keys():
            dict_sid2span_salient[sid].append(span)
        else:
            dict_sid2span_salient[sid] = [span]

    dict_sid2span_salient_pred = {}
    for span in spans_pred:
        sid = dict_span2sid_pred[span]
        if sid in dict_sid2span_salient_pred.keys():
            dict_sid2span_salient_pred[sid].append(span)
        else:
            dict_sid2span_salient_pred[sid] = [span]

    performance_list = []
    confidence_low, confidence_up = 0, 0
    for i in range(n_times):
        sample_index_list = choices(range(n_data), k=n_sampling)

        true_label_bootstrap_list = []
        pred_label_bootstrap_list = []
        for ind, sid in enumerate(sample_index_list):

            if sid in dict_sid2span_salient.keys():
                true_label_list = dict_sid2span_salient[sid]
                true_label_list_revised = [true_label + "_" + str(ind) for true_label in true_label_list]
                true_label_bootstrap_list += true_label_list_revised

            if sid in dict_sid2span_salient_pred.keys():
                pred_label_list = dict_sid2span_salient_pred[sid]
                pred_label_list_revised = [pred_label + "_" + str(ind) for pred_label in pred_label_list]
                pred_label_bootstrap_list += pred_label_list_revised

        f1, p, r = ea.evaluate_chunk_level(pred_label_bootstrap_list, true_label_bootstrap_list)
        performance_list.append(f1)

    if n_times != 1000:
        confidence_low, confidence_up = ea.mean_confidence_interval(performance_list)
    else:
        performance_list.sort()
        confidence_low = performance_list[24]
        confidence_up = performance_list[974]

    # print("\n")
    # print("confidence_low:\t", confidence_low)
    # print("confidence_up:\t", confidence_up)

    return confidence_low, confidence_up


def get_bucket_f1(dict_bucket2span, dict_bucket2span_pred, dict_span2sid, dict_span2sid_pred, dict_chunkid2span,
                  dict_chunkid2span_pred, is_print_ci, is_print_case):
    error_case_list = []

    dict_bucket2f1 = {}

    # predict:  2_3 -> NER
    dict_pos2tag_pred = {}
    if is_print_case:
        for k_bucket_eval, spans_pred in dict_bucket2span_pred.items():
            for span_pred in spans_pred:
                pos_pred = "_".join(span_pred.split("_")[0:2])
                tag_pred = span_pred.split("_")[-1]
                dict_pos2tag_pred[pos_pred] = tag_pred
    # print(dict_pos2tag_pred)

    # true:  2_3 -> NER
    dict_pos2tag = {}
    if is_print_case:
        for k_bucket_eval, spans in dict_bucket2span.items():
            for span in spans:
                pos = "_".join(span.split("_")[0:2])
                tag = span.split("_")[-1]
                dict_pos2tag[pos] = tag
    # print(dict_pos2tag_pred)

    if is_print_case:
        error_case_list = get_error_case(dict_pos2tag, dict_pos2tag_pred, dict_chunkid2span, dict_chunkid2span_pred)

    for bucket_interval, spans_true in dict_bucket2span.items():
        spans_pred = []

        # print('bucket_interval: ',bucket_interval)
        if bucket_interval not in dict_bucket2span_pred.keys():
            # print(bucket_interval)
            raise ValueError("Predict Label Bucketing Errors")
        else:
            spans_pred = dict_bucket2span_pred[bucket_interval]

        confidence_low, confidence_up = 0, 0

        if is_print_ci:
            confidence_low, confidence_up = compute_confidence_interval_f1(spans_true, spans_pred, dict_span2sid,
                                                                           dict_span2sid_pred)

        confidence_low = format(confidence_low, '.3g')
        confidence_up = format(confidence_up, '.3g')

        f1, p, r = ea.evaluate_chunk_level(spans_pred, spans_true)

        error_entity_list = []
        if is_print_case:
            for span_true in spans_true:
                if span_true not in spans_pred:
                    # print(span_true)
                    pos_true = "_".join(span_true.split("_")[0:2])
                    tag_true = span_true.split("_")[-1]

                    if pos_true in dict_pos2tag_pred.keys():
                        tag_pred = dict_pos2tag_pred[pos_true]
                        if tag_pred != tag_true:
                            error_entity_list.append(
                                ea.format4json2(dict_chunkid2span[span_true]) + "|||" + tag_true + "|||" +
                                dict_pos2tag_pred[pos_true])
                    else:
                        # error_entity_list.append(format4json_tc(dict_chunkid2span[span_true]) + "|||" + tag_true + "|||" + "O")
                        continue

        # print("confidence_low:\t", confidence_low)
        # print("confidence_up:\t", confidence_up)
        # print("F1:\t", f1)
        # print(error_entity_list)

        dict_bucket2f1[bucket_interval] = [f1, len(spans_true), confidence_low, confidence_up, error_entity_list]

    # if bucket_interval[0] == 1.0:
    # 	print("debug-f1:",f1)
    # 	print(spans_pred[0:20])
    # 	print(spans_true[0:20])
    # print("dict_bucket2f1: ",dict_bucket2f1)
    return ea.sort_dict(dict_bucket2f1), error_case_list