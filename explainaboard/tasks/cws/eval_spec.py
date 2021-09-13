from random import choices

import explainaboard.error_analysis as ea
import numpy
import pickle
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

        # tag='B-'+tag
        tag = tag + "-W"
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


def get_aspect_value(test_word_sequences, test_true_tag_sequences, test_word_sequences_sent,
                   test_true_tag_sequences_sent, dict_precomputed_path, dict_aspect_func):
    def get_sentential_value(test_true_tag_sequences_sent, test_word_sequences_sent):

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
    for aspect, fun in dict_aspect_func.items():
        dict_span2aspect_val[aspect] = {}

    eDen_list, sentLen_list = get_sentential_value(test_true_tag_sequences_sent,
                                                   test_word_sequences_sent)

    dict_pos2sid = ea.get_pos2sentid(test_word_sequences_sent)
    dict_ap2rp = ea.get_token_position(test_word_sequences_sent)
    all_chunks = ea.get_chunks(test_true_tag_sequences)

    dict_span2sid = {}
    dict_chunkid2span = {}
    for span_info in all_chunks:

        # print(span_info)

        # span_type = span_info[0].lower()

        # print(span_type)
        idx_start = span_info[1]
        idx_end = span_info[2]
        span_cnt = ''.join(test_word_sequences[idx_start:idx_end]).lower()
        # print(span_cnt.encode("utf-8").decode("utf-8"))
        span_cnt = span_cnt.encode("gbk", "ignore").decode("gbk", "ignore")
        # print(sys.getdefaultencoding())
        span_type = ''.join(test_true_tag_sequences[idx_start:idx_end])

        span_pos = str(idx_start) + "|||" + str(idx_end) + "|||" + span_type

        if len(span_type) != (idx_end - idx_start):
            print(idx_start, idx_end)
            print(span_info)
            print(span_type + "\t" + span_cnt)
            print("--------------")

        # print(span_pos)
        # print(span_info)
        # print(span_cnt)

        span_length = idx_end - idx_start

        # span_token_list = test_word_sequences[idx_start:idx_end]
        # span_token_pos_list = [str(pos) + "|||" + span_type for pos in range(idx_start, idx_end)]
        # print(span_token_pos_list)

        span_sentid = dict_pos2sid[idx_start]
        sLen = float(sentLen_list[span_sentid])

        dict_span2sid[span_pos] = span_sentid

        text_sample = "".join(test_word_sequences_sent[span_sentid])
        text_sample = text_sample

        dict_chunkid2span[span_pos] = span_cnt + "|||" + text_sample

        # Sentence Length: sLen
        aspect = "sLen"
        if aspect in dict_aspect_func.keys():
            dict_span2aspect_val[aspect][span_pos] = sLen

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
        confidence_low_overall, confidence_up_overall = compute_confidence_interval_f1(dict_span2sid.keys(),
                                                                                       dict_span2sid_pred.keys(),
                                                                                       dict_span2sid,
                                                                                       dict_span2sid_pred,
                                                                                       n_times=10)

    print("confidence_low_overall:\t", confidence_low_overall)
    print("confidence_up_overall:\t", confidence_up_overall)

    print("------------------ Holistic Result")
    print(holistic_performance)

    # print(f1(list_true_tags_token, list_pred_tags_token)["f1"])

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
                                                                dict_chunkid2span_pred, list_true_tags_token,
                                                                list_pred_tags_token, is_print_ci, is_print_case)
        aspect_names.append(aspect)
    print("aspect_names: ", aspect_names)

    # for v in error_case_list:
    # 	print(v)

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
                                             "bucket_error_case": error_entity_list[
                                                                  0:int(len(error_entity_list) / 10)]})

    obj_json["task"] = task_type
    obj_json["data"]["language"] = "Chinese"
    obj_json["data"]["bias"] = dict_aspect2bias

    obj_json["model"]["results"]["overall"]["performance"] = holistic_performance
    obj_json["model"]["results"]["overall"]["confidence_low"] = confidence_low_overall
    obj_json["model"]["results"]["overall"]["confidence_up"] = confidence_up_overall
    obj_json["model"]["results"]["fine_grained"] = dict_fine_grained

    # Save error cases: overall
    obj_json["model"]["results"]["overall"]["error_case"] = error_case_list[0:int(len(error_case_list) / 10)]

    ea.save_json(obj_json, output_filename)


def compute_confidence_interval_f1(spans_true, spans_pred, dict_span2sid, dict_span2sid_pred, n_times=1000):
    n_data = len(dict_span2sid)
    sample_rate = ea.get_sample_rate(n_data)
    n_sampling = int(n_data * sample_rate)
    print("sample_rate:\t", sample_rate)
    print("n_sampling:\t", n_sampling)

    dict_sid2span_salient = {}
    for span in spans_true:
        # print(span)
        if len(span.split("|||")) != 3:
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
                true_label_list_revised = [true_label + "|||" + str(ind) for true_label in true_label_list]
                true_label_bootstrap_list += true_label_list_revised

            if sid in dict_sid2span_salient_pred.keys():
                pred_label_list = dict_sid2span_salient_pred[sid]
                pred_label_list_revised = [pred_label + "|||" + str(ind) for pred_label in pred_label_list]
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


def get_error_case(dict_pos2tag, dict_pos2tag_pred, dict_chunkid2span_sent, dict_chunkid2span_sent_pred,
                   list_true_tags_token, list_pred_tags_token):
    error_case_list = []
    for pos, tag in dict_pos2tag.items():

        true_label = tag
        pred_label = ""
        # print(dict_chunkid2span_sent.keys())
        if pos + "|||" + tag not in dict_chunkid2span_sent.keys():
            continue
        span_sentence = dict_chunkid2span_sent[pos + "|||" + tag]

        if pos in dict_pos2tag_pred.keys():
            pred_label = dict_pos2tag_pred[pos]
            if true_label == pred_label:
                continue
        # print(pos + "\t" + true_label + "\t" + pred_label)
        else:
            start = int(pos.split("|||")[0])
            end = int(pos.split("|||")[1])
            pred_label = "".join(list_pred_tags_token[start:end])
        # print(pred_label)
        error_case = span_sentence + "|||" + true_label + "|||" + pred_label
        error_case_list.append(error_case)

    for pos, tag in dict_pos2tag_pred.items():

        true_label = ""
        pred_label = tag
        if pos + "|||" + tag not in dict_chunkid2span_sent_pred.keys():
            continue
        span_sentence = dict_chunkid2span_sent_pred[pos + "|||" + tag]

        if pos in dict_pos2tag.keys():
            true_label = dict_pos2tag[pos]
            if true_label == pred_label:
                continue
        else:
            start = int(pos.split("|||")[0])
            end = int(pos.split("|||")[1])
            true_label = "".join(list_true_tags_token[start:end])
        error_case = span_sentence + "|||" + true_label + "|||" + pred_label
        error_case_list.append(error_case)

    # for v in error_case_list:
    # 	print(len(error_case_list))
    # 	print(v)
    # print(error_case_list)

    return error_case_list


def get_bucket_f1(dict_bucket2span, dict_bucket2span_pred, dict_span2sid, dict_span2sid_pred, dict_chunkid2span,
                  dict_chunkid2span_pred, list_true_tags_token, list_pred_tags_token, is_print_ci, is_print_case):
    dict_bucket2f1 = {}

    # predict:  2_3 -> NER
    dict_pos2tag_pred = {}
    if is_print_case:
        for k_bucket_eval, spans_pred in dict_bucket2span_pred.items():
            for span_pred in spans_pred:
                pos_pred = "|||".join(span_pred.split("|||")[0:2])
                tag_pred = span_pred.split("|||")[-1]
                dict_pos2tag_pred[pos_pred] = tag_pred

    # true:  2_3 -> NER
    dict_pos2tag = {}
    if is_print_case:
        for k_bucket_eval, spans in dict_bucket2span.items():
            for span in spans:
                pos = "|||".join(span.split("|||")[0:2])
                tag = span.split("|||")[-1]
                dict_pos2tag[pos] = tag

    error_case_list = []
    if is_print_case:
        error_case_list = get_error_case(dict_pos2tag, dict_pos2tag_pred, dict_chunkid2span, dict_chunkid2span_pred,
                                         list_true_tags_token, list_pred_tags_token)

    # print(len(error_case_list))
    # print(error_case_list)

    for bucket_interval, spans_true in dict_bucket2span.items():
        spans_pred = []
        if bucket_interval not in dict_bucket2span_pred.keys():
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
                    pos_true = "|||".join(span_true.split("|||")[0:2])
                    tag_true = span_true.split("|||")[-1]

                    if pos_true in dict_pos2tag_pred.keys():
                        tag_pred = dict_pos2tag_pred[pos_true]
                        if tag_pred != tag_true:
                            error_entity_list.append(
                                dict_chunkid2span[span_true] + "|||" + tag_true + "|||" + dict_pos2tag_pred[pos_true])
                    # print(dict_chunkid2span[span_true] + "|||" + tag_true + "|||" + dict_pos2tag_pred[pos_true])
                    else:
                        start = int(pos_true.split("|||")[0])
                        end = int(pos_true.split("|||")[1])
                        pred_label = "".join(list_pred_tags_token[start:end])
                        error_entity_list.append(dict_chunkid2span[span_true] + "|||" + tag_true + "|||" + pred_label)

            # print(dict_chunkid2span[span_true] + "|||" + tag_true + "|||" + pred_label)

        dict_bucket2f1[bucket_interval] = [f1, len(spans_true), confidence_low, confidence_up, error_entity_list]

    # if bucket_interval[0] == 1.0:
    # 	print("debug-f1:",f1)
    # 	print(spans_pred[0:20])
    # 	print(spans_true[0:20])
    # print("dict_bucket2f1: ",dict_bucket2f1)
    return ea.sort_dict(dict_bucket2f1), error_case_list