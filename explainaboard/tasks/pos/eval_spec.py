# -*- coding: utf-8 -*-
import explainaboard.error_analysis as ea
import pickle
import numpy


def get_aspect_value(test_word_sequences, test_trueTag_sequences, test_word_sequences_sent,
                   test_trueTag_sequences_sent, dict_precomputed_path, dict_aspect_func):
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

        token_type = test_trueTag_sequences[token_id]
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


def evaluate(task_type="ner", analysis_type="single", systems=[], output="./output.json", is_print_ci=False,
             is_print_case=False, is_print_ece=False):
    path_text = ""

    if analysis_type == "single":
        path_text = systems[0]

    corpus_type = "dataset_name"
    model_name = "model_name"
    path_precomputed = ""
    path_aspect_conf = "./explainaboard/tasks/pos/conf.aspects"
    path_json_input = "./explainaboard/tasks/pos/template.json"
    fn_write_json = output

    # Initalization
    dict_aspect_func = ea.load_conf(path_aspect_conf)
    metric_names = list(dict_aspect_func.keys())
    print("dict_aspect_func: ", dict_aspect_func)
    print(dict_aspect_func)

    fwrite_json = open(fn_write_json, 'w')
    path_comb_output = model_name + "/" + path_text.split("/")[-1]
    # get precomputed paths from conf file
    dict_precomputed_path = {}
    for aspect, func in dict_aspect_func.items():
        is_precomputed = func[2].lower()
        if is_precomputed == "yes":
            dict_precomputed_path[aspect] = path_precomputed + "_" + aspect + ".pkl"
            print("precomputed directory:\t", dict_precomputed_path[aspect])

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

    def __selectBucktingFunc(func_name, func_setting, dict_obj):
        if func_name == "bucket_attribute_SpecifiedBucketInterval":
            return ea.bucket_attribute_specified_bucket_interval(dict_obj, eval(func_setting))
        elif func_name == "bucket_attribute_SpecifiedBucketValue":
            if len(func_setting.split("\t")) != 2:
                raise ValueError("selectBucktingFunc Error!")
            n_buckets, specified_bucket_value_list = int(func_setting.split("\t")[0]), eval(func_setting.split("\t")[1])
            return ea.bucket_attribute_specified_bucket_value(dict_obj, n_buckets, specified_bucket_value_list)
        elif func_name == "bucket_attribute_DiscreteValue":  # now the discrete value is R-tag..
            if len(func_setting.split("\t")) != 2:
                raise ValueError("selectBucktingFunc Error!")
            tags_list = list(set(dict_obj.values()))
            topK_buckets, min_buckets = int(func_setting.split("\t")[0]), int(func_setting.split("\t")[1])
            # return eval(func_name)(dict_obj, min_buckets, topK_buckets)
            return ea.bucket_attribute_discrete_value(dict_obj, topK_buckets, min_buckets)
        else:
            raise ValueError(f'Illegal function name {func_name}')

    dict_bucket2span = {}
    dict_bucket2span_pred = {}
    dict_bucket2f1 = {}
    aspect_names = []
    error_case_list = []

    for aspect, func in dict_aspect_func.items():
        # print(aspect, dict_span2aspect_val[aspect])
        dict_bucket2span[aspect] = __selectBucktingFunc(func[0], func[1], dict_span2aspect_val[aspect])
        # print(aspect, dict_bucket2span[aspect])
        # exit()
        dict_bucket2span_pred[aspect] = ea.bucket_attribute_specified_bucket_interval(dict_span2aspect_val_pred[aspect],
                                                                                      dict_bucket2span[aspect].keys())
        dict_bucket2f1[aspect], error_case_list = ea.getBucketF1_pos(dict_bucket2span[aspect],
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

    def beautifyInterval(interval):

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

    dict_fineGrained = {}
    for aspect, metadata in dict_bucket2f1.items():
        dict_fineGrained[aspect] = []
        for bucket_name, v in metadata.items():
            # print("---------debug--bucket name old---")
            # print(bucket_name)
            bucket_name = beautifyInterval(bucket_name)
            # print("---------debug--bucket name new---")
            # print(bucket_name)

            # bucket_value = format(v[0]*100,'.4g')
            bucket_value = format(float(v[0]) * 100, '.4g')
            n_sample = v[1]
            confidence_low = format(float(v[2]) * 100, '.4g')
            confidence_up = format(float(v[3]) * 100, '.4g')
            error_entity_list = v[4]

            # instantiation
            dict_fineGrained[aspect].append({"bucket_name": bucket_name, "bucket_value": bucket_value, "num": n_sample,
                                             "confidence_low": confidence_low, "confidence_up": confidence_up,
                                             "bucket_error_case": error_entity_list})

    obj_json = ea.load_json(path_json_input)

    obj_json["task"] = task_type
    obj_json["data"]["name"] = corpus_type
    obj_json["data"]["language"] = "English"
    obj_json["data"]["bias"] = dict_aspect2bias
    obj_json["data"]["output"] = path_comb_output
    obj_json["model"]["name"] = model_name
    obj_json["model"]["results"]["overall"]["performance"] = holistic_performance
    obj_json["model"]["results"]["overall"]["confidence_low"] = confidence_low_overall
    obj_json["model"]["results"]["overall"]["confidence_up"] = confidence_up_overall
    obj_json["model"]["results"]["fine_grained"] = dict_fineGrained

    obj_json["model"]["results"]["overall"]["error_case"] = error_case_list

    ea.save_json(obj_json, fn_write_json)
