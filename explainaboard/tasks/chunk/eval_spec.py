# -*- coding: utf-8 -*-
import explainaboard.error_analysis as ea
import pickle
import numpy
import os


def get_aspect_value(test_word_sequences, test_true_tag_sequences, test_word_sequences_sent,
                   test_true_tag_sequences_sent, dict_precomputed_path, dict_aspect_func):
    def getSententialValue(test_true_tag_sequences_sent, test_word_sequences_sent):

        eDen = []
        sentLen = []

        for i, test_sent in enumerate(test_true_tag_sequences_sent):
            pred_chunks = set(ea.get_chunks(test_sent))

            num_entityToken = 0
            for pred_chunk in pred_chunks:
                idx_start = pred_chunk[1]
                idx_end = pred_chunk[2]
                num_entityToken += idx_end - idx_start

            # introduce the entity token density in sentence ...
            eDen.append(float(num_entityToken) / len(test_sent))

            # introduce the sentence length in sentence ...
            sentLen.append(len(test_sent))

        return eDen, sentLen

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
    dict_chunkid2span = {}
    for aspect, fun in dict_aspect_func.items():
        dict_span2aspect_val[aspect] = {}

    eDen_list, sentLen_list = [], []
    eDen_list, sentLen_list = getSententialValue(test_true_tag_sequences_sent,
                                                 test_word_sequences_sent)

    dict_pos2sid = ea.get_pos2sentid(test_word_sequences_sent)
    dict_ap2rp = ea.get_token_position(test_word_sequences_sent)
    all_chunks = ea.get_chunks(test_true_tag_sequences)
    dict_span2sid = {}
    for span_info in all_chunks:

        span_type = span_info[0].lower()
        # print(span_type)
        idx_start = span_info[1]
        idx_end = span_info[2]
        span_cnt = ' '.join(test_word_sequences[idx_start:idx_end]).lower()
        span_pos = str(idx_start) + "_" + str(idx_end) + "_" + span_type

        span_length = idx_end - idx_start

        span_token_list = test_word_sequences[idx_start:idx_end]
        span_token_pos_list = [str(pos) + "_" + span_type for pos in range(idx_start, idx_end)]

        span_sentid = dict_pos2sid[idx_start]

        sLen = float(sentLen_list[span_sentid])

        dict_span2sid[span_pos] = span_sentid
        dict_chunkid2span[span_pos] = ea.format4json(span_cnt) + "|||" + ea.format4json(
            ' '.join(test_word_sequences_sent[span_sentid]))

        # Sentence Length: sLen
        aspect = "sLen"
        if aspect in dict_aspect_func.keys():
            dict_span2aspect_val[aspect][span_pos] = sLen

        # Relative Position: relPos
        aspect = "rPos"
        if aspect in dict_aspect_func.keys():
            dict_span2aspect_val[aspect][span_pos] = (dict_ap2rp[idx_start]) * 1.0 / sLen

        # Entity Length: eLen
        aspect = "eLen"
        if aspect in dict_aspect_func.keys():
            dict_span2aspect_val[aspect][span_pos] = float(span_length)

        # Tag: tag
        aspect = "tag"
        if aspect in dict_aspect_func.keys():
            dict_span2aspect_val[aspect][span_pos] = span_type

    # print(dict_span2aspect_val)
    return dict_span2aspect_val, dict_span2sid, dict_chunkid2span


def tuple2str(triplet):
    res = ""
    for v in triplet:
        res += str(v) + "_"
    return res.rstrip("_")


def evaluate(task_type="ner", analysis_type="single", systems=[], output_filename="./output.json", is_print_ci=False,
             is_print_case=False, is_print_ece=False):

    path_text = systems[0] if analysis_type == "single" else ""
    path_comb_output = "model_name" + "/" + path_text.split("/")[-1]
    dict_aspect_func, dict_precomputed_path, obj_json = ea.load_task_conf(task_dir=os.path.dirname(__file__))

    list_text_sent, list_text_token = ea.read_single_column(path_text, 0)
    list_true_tags_sent, list_true_tags_token = ea.read_single_column(path_text, 1)
    list_pred_tags_sent, list_pred_tags_token = ea.read_single_column(path_text, 2)

    dict_span2aspect_val, dict_span2sid, dict_chunkid2span = get_aspect_value(list_text_token, list_true_tags_token,
                                                                           list_text_sent, list_true_tags_sent,
                                                                           dict_precomputed_path, dict_aspect_func)
    dict_span2aspect_val_pred, dict_span2sid_pred, dict_chunkid2span_pred = get_aspect_value(list_text_token,
                                                                                          list_pred_tags_token,
                                                                                          list_text_sent,
                                                                                          list_pred_tags_sent,
                                                                                          dict_precomputed_path,
                                                                                          dict_aspect_func)

    holistic_performance = ea.f1(list_true_tags_sent, list_pred_tags_sent)["f1"]
    confidence_low_overall, confidence_up_overall = 0, 0
    if is_print_ci:
        confidence_low_overall, confidence_up_overall = ea.compute_confidence_interval_f1(dict_span2sid.keys(),
                                                                                          dict_span2sid_pred.keys(),
                                                                                          dict_span2sid,
                                                                                          dict_span2sid_pred,
                                                                                          n_times=1000)

    # print(dict_span2aspect_val)

    print("confidence_low_overall:\t", confidence_low_overall)
    print("confidence_up_overall:\t", confidence_up_overall)
    # holistic_performance = f1(list_true_tags_sent, list_pred_tags_sent)["f1"]
    # print(f1(list_true_tags_sent, list_pred_tags_sent))

    print("------------------ Holistic Result")
    print(holistic_performance)

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
        dict_bucket2f1[aspect], error_case_list = get_bucket_f1_chunk(dict_bucket2span[aspect],
                                                                      dict_bucket2span_pred[aspect], dict_span2sid,
                                                                      dict_span2sid_pred, dict_chunkid2span,
                                                                      dict_chunkid2span_pred, is_print_ci,
                                                                      is_print_case)
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

    ea.save_json(obj_json, "./instantiate.json")
    ea.save_json(obj_json, output_filename)


def get_bucket_f1_chunk(dict_bucket2span, dict_bucket2span_pred, dict_span2sid, dict_span2sid_pred, dict_chunkid2span,
                      dict_chunkid2span_pred, is_print_ci, is_print_case):
    dict_bucket2f1 = {}

    # predict:  2_3 -> NER
    dict_pos2tag_pred = {}
    if is_print_case:
        for k_bucket_eval, spans_pred in dict_bucket2span_pred.items():
            for span_pred in spans_pred:
                pos_pred = "_".join(span_pred.split("_")[0:2])
                tag_pred = span_pred.split("_")[-1]
                dict_pos2tag_pred[pos_pred] = tag_pred

    # true:  2_3 -> NER
    dict_pos2tag = {}
    if is_print_case:
        for k_bucket_eval, spans in dict_bucket2span.items():
            for span in spans:
                pos = "_".join(span.split("_")[0:2])
                tag = span.split("_")[-1]
                dict_pos2tag[pos] = tag

    error_case_list = []
    if is_print_case:
        error_case_list = ea.get_error_case(dict_pos2tag, dict_pos2tag_pred, dict_chunkid2span, dict_chunkid2span_pred)

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
            confidence_low, confidence_up = ea.compute_confidence_interval_f1(spans_true, spans_pred, dict_span2sid,
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
                                dict_chunkid2span[span_true] + "|||" + tag_true + "|||" + dict_pos2tag_pred[pos_true])
                    else:
                        error_entity_list.append(dict_chunkid2span[span_true] + "|||" + tag_true + "|||" + "O")

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