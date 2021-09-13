# -*- coding: utf-8 -*-
import explainaboard.error_analysis as ea
import pickle
import numpy
import codecs
import os

def read_data(corpus_type, fn, column_no=-1, delimiter=' '):
    print('corpus_type', corpus_type)
    word_sequences = list()
    tag_sequences = list()
    total_word_sequences = list()
    total_tag_sequences = list()
    with codecs.open(fn, 'r', 'utf-8') as f:
        lines = f.readlines()
    curr_words = list()
    curr_tags = list()
    for k in range(len(lines)):
        line = lines[k].strip()
        if len(line) == 0 or line.startswith('-DOCSTART-'):  # new sentence or new document
            if len(curr_words) > 0:
                word_sequences.append(curr_words)
                tag_sequences.append(curr_tags)
                curr_words = list()
                curr_tags = list()
            continue

        strings = line.split(delimiter)
        word = strings[0].strip()
        tag = strings[column_no].strip()  # be default, we take the last tag

        tag = 'B-' + tag
        curr_words.append(word)
        curr_tags.append(tag)
        total_word_sequences.append(word)
        total_tag_sequences.append(tag)
        if k == len(lines) - 1:
            word_sequences.append(curr_words)
            tag_sequences.append(curr_tags)
    # if verbose:
    # 	print('Loading from %s: %d samples, %d words.' % (fn, len(word_sequences), get_words_num(word_sequences)))
    # return word_sequences, tag_sequences
    return total_word_sequences, total_tag_sequences, word_sequences, tag_sequences


#   get_aspect_value(test_word_sequences, test_true_tag_sequences, test_word_sequences_sent, dict_precomputed_path)

def get_aspect_value(test_word_sequences, test_true_tag_sequences, test_word_sequences_sent,
                   test_true_tag_sequences_sent, dict_precomputed_path, dict_aspect_func):
    def getSententialValue(test_true_tag_sequences_sent, test_word_sequences_sent, dict_oov=None):

        eDen = []
        sentLen = []
        oDen = []

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

            # introduce the oov density in sentence ...
            if dict_oov != None:
                num_oov = 0
                for word in test_word_sequences_sent[i]:
                    if word not in dict_oov:
                        num_oov += 1
                oDen.append(float(num_oov) / len(test_sent))

        return eDen, sentLen, oDen

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

    eDen_list, sentLen_list = [], []
    dict_oov = None
    if "oDen" in dict_precomputed_model.keys():
        dict_oov = dict_precomputed_model['oDen']

    eDen_list, sentLen_list, oDen_list = getSententialValue(test_true_tag_sequences_sent,
                                                            test_word_sequences_sent, dict_oov)

    # print(oDen_list)

    dict_pos2sid = ea.get_pos2sentid(test_word_sequences_sent)
    dict_ap2rp = ea.get_token_position(test_word_sequences_sent)
    all_chunks = ea.get_chunks(test_true_tag_sequences)

    dict_span2sid = {}
    dict_chunkid2span = {}
    for span_info in all_chunks:

        span_type = span_info[0].lower()

        idx_start = span_info[1]
        idx_end = span_info[2]
        span_sentid = dict_pos2sid[idx_start]
        span_cnt = ' '.join(test_word_sequences[idx_start:idx_end])

        span_pos = str(idx_start) + "_" + str(idx_end) + "_" + span_type

        # if str(idx_start) != "" or str(idx_end)!= "":

        span_length = idx_end - idx_start

        dict_span2sid[span_pos] = span_sentid

        dict_chunkid2span[span_pos] = ea.format4json(span_cnt) + "|||" + ea.format4json(
            ' '.join(test_word_sequences_sent[span_sentid]))
        # print(dict_chunkid2span[span_pos])
        # dict_chunkid2span[span_pos] = ' '.join(test_word_sequences[idx_start:idx_end])
        # for bootstrapping
        # if span_sentid not in dict_sid2span.keys():
        # 	dict_sid2span[span_sentid] = [span_pos]
        # else:
        # 	dict_sid2span[span_sentid].append(span_pos)

        span_token_list = test_word_sequences[idx_start:idx_end]
        span_token_pos_list = [str(pos) + "_" + span_type for pos in range(idx_start, idx_end)]

        sLen = float(sentLen_list[span_sentid])

        # Sentence Length: sLen
        aspect = "sLen"
        if aspect in dict_aspect_func.keys():
            dict_span2aspect_val[aspect][span_pos] = sLen
        #
        #
        # # Relative Position: relPos
        aspect = "rPos"
        if aspect in dict_aspect_func.keys():
            dict_span2aspect_val[aspect][span_pos] = (dict_ap2rp[idx_start]) * 1.0 / sLen
        #
        #
        # # Entity Length: eLen
        aspect = "eLen"
        if aspect in dict_aspect_func.keys():
            dict_span2aspect_val[aspect][span_pos] = float(span_length)
        #
        # # Entity Density: eDen
        aspect = "eDen"
        if aspect in dict_aspect_func.keys():
            dict_span2aspect_val[aspect][span_pos] = float(eDen_list[span_sentid])
        #
        #
        #
        # # Tag: tag
        aspect = "tag"
        if aspect in dict_aspect_func.keys():
            dict_span2aspect_val[aspect][span_pos] = span_type
        #
        #
        # # Tag: tag
        aspect = "capital"
        if aspect in dict_aspect_func.keys():
            dict_span2aspect_val[aspect][span_pos] = ea.cap_feature(span_cnt)

        # OOV Density: oDen
        aspect = "oDen"
        if aspect in dict_aspect_func.keys():
            dict_span2aspect_val[aspect][span_pos] = float(oDen_list[span_sentid])

        # Span-level Frequency: fre_span
        aspect = "eFre"
        span_cnt_lower = span_cnt.lower()
        if aspect in dict_aspect_func.keys():
            preCompute_freqSpan = dict_precomputed_model[aspect]
            span_fre_value = 0.0
            if span_cnt_lower in preCompute_freqSpan:
                span_fre_value = preCompute_freqSpan[span_cnt_lower]
            dict_span2aspect_val[aspect][span_pos] = float(span_fre_value)
        # dict_span2sid[aspect][span_pos] = span_sentid

        aspect = "eCon"
        if aspect in dict_aspect_func.keys():
            preCompute_ambSpan = dict_precomputed_model[aspect]
            span_amb_value = 0.0
            if span_cnt_lower in preCompute_ambSpan:
                if span_type.lower() in preCompute_ambSpan[span_cnt_lower]:
                    span_amb_value = preCompute_ambSpan[span_cnt_lower][span_type]
            dict_span2aspect_val[aspect][span_pos] = span_amb_value

    # print(dict_chunkid2span)
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

    # Confidence Interval of Holistic Performance
    confidence_low_overall, confidence_up_overall = 0, 0
    if is_print_ci:
        confidence_low_overall, confidence_up_overall = ea.compute_confidence_interval_f1(dict_span2sid.keys(),
                                                                                          dict_span2sid_pred.keys(),
                                                                                          dict_span2sid,
                                                                                          dict_span2sid_pred,
                                                                                          n_times=100)

    print("confidence_low_overall:\t", confidence_low_overall)
    print("confidence_up_overall:\t", confidence_up_overall)

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

    # dict_fine_grained[aspect].append({"bucket_name":bucket_name, "bucket_value":bucket_value, "num":n_sample, "confidence_low":confidence_low, "confidence_up":confidence_up, "bucket_error_case":[]})

    obj_json["task"] = task_type
    obj_json["data"]["output"] = path_comb_output
    obj_json["data"]["language"] = "English"
    obj_json["data"]["bias"] = dict_aspect2bias

    # obj_json["model"]["results"]["overall"]["error_case"] = []
    obj_json["model"]["results"]["overall"]["error_case"] = error_case_list
    obj_json["model"]["results"]["overall"]["performance"] = holistic_performance
    obj_json["model"]["results"]["overall"]["confidence_low"] = confidence_low_overall
    obj_json["model"]["results"]["overall"]["confidence_up"] = confidence_up_overall
    obj_json["model"]["results"]["fine_grained"] = dict_fine_grained

    ea.save_json(obj_json, output_filename)


def get_bucket_f1(dict_bucket2span, dict_bucket2span_pred, dict_span2sid, dict_span2sid_pred, dict_chunkid2span,
                  dict_chunkid2span_pred, is_print_ci, is_print_case):
    # print('------------------ attribute')
    dict_bucket2f1 = {}

    # predict:  2_3 -> NER
    dict_pos2tag_pred = {}
    for k_bucket_eval, spans_pred in dict_bucket2span_pred.items():
        for span_pred in spans_pred:
            pos_pred = "_".join(span_pred.split("_")[0:2])
            tag_pred = span_pred.split("_")[-1]
            dict_pos2tag_pred[pos_pred] = tag_pred
        # print(dict_pos2tag_pred)

    # true:  2_3 -> NER
    dict_pos2tag = {}
    for k_bucket_eval, spans in dict_bucket2span.items():
        for span in spans:
            pos = "_".join(span.split("_")[0:2])
            tag = span.split("_")[-1]
            dict_pos2tag[pos] = tag
    # print(dict_pos2tag_pred)

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

        # print("-----------print spans_pred -------------")

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

            # print("------------------------------------------")

        dict_bucket2f1[bucket_interval] = [f1, len(spans_true), confidence_low, confidence_up, error_entity_list]

        # if bucket_interval[0] == 1.0:
        # 	print("debug-f1:",f1)
        # 	print(spans_pred[0:20])
        # 	print(spans_true[0:20])
    # print("dict_bucket2f1: ",dict_bucket2f1)
    return ea.sort_dict(dict_bucket2f1), error_case_list