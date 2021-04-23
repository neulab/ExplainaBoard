# -*- coding: utf-8 -*-
import argparse
import numpy
import sys
from sklearn.metrics import f1_score

sys.path.append("../src")
from errorAnalysis import *
from utils import *


#   getAspectValue(test_word_sequences, test_trueTag_sequences, test_word_sequences_sent, dict_precomputed_path)

def getAspectValue(test_word_sequences, test_trueTag_sequences, test_word_sequences_sent,
                   test_trueTag_sequences_sent, dict_preComputed_path, dict_aspect_func):
    dict_preComputed_model = {}
    for aspect, path in dict_preComputed_path.items():
        print("path:\t" + path)
        if os.path.exists(path):
            print('load the hard dictionary of entity span in test set...')
            fread = open(path, 'rb')
            dict_preComputed_model[aspect] = pickle.load(fread)
        else:
            raise ValueError("can not load hard dictionary" + aspect + "\t" + path)

    dict_span2aspectVal = {}
    for aspect, fun in dict_aspect_func.items():
        dict_span2aspectVal[aspect] = {}

    dict_pos2sid = getPos2SentId(test_word_sequences_sent)
    dict_ap2rp = getTokenPosition(test_word_sequences_sent)


    dict_span2sid = {}
    dict_chunkid2span = {}
    for token_id, token in enumerate(test_word_sequences):

        token_type = test_trueTag_sequences[token_id]
        token_pos = str(token_id) + "_" + str(token) + "_" + token_type
        token_sentid = dict_pos2sid[token_id]
        sLen = float(len(test_word_sequences_sent[token_sentid]))




        dict_span2sid[token_pos] = token_sentid
        dict_chunkid2span[token_pos] = token + "|||" + format4json(' '.join(test_word_sequences_sent[token_sentid]))

        # Sentence Length: sentLen
        aspect = "sLen"
        if aspect in dict_aspect_func.keys():
            dict_span2aspectVal[aspect][token_pos] = sLen

        # Sentence Length: tokLen
        aspect = "tLen"
        if aspect in dict_aspect_func.keys():
            dict_span2aspectVal[aspect][token_pos] = float(len(token))
            # if float(len(token)) == 1:
            #     print(token)

        # Relative Position: relPos
        aspect = "rPos"
        if aspect in dict_aspect_func.keys():
            dict_span2aspectVal[aspect][token_pos] = (dict_ap2rp[token_id]) * 1.0 / sLen

        # Tag: tag
        aspect = "tag"
        if aspect in dict_aspect_func.keys():
            dict_span2aspectVal[aspect][token_pos] = token_type

    return dict_span2aspectVal, dict_span2sid, dict_chunkid2span


# def tuple2str(triplet):
#     res = ""
#     for v in triplet:
#         res += str(v) + "_"
#     return res.rstrip("_")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learning tagger using neural networks')

    parser.add_argument('--text', type=str, required=False,
                        help="the type of the task")

    parser.add_argument('--true_file', type=str, required=False,
                        help="model's name")

    parser.add_argument('--pred_file', type=str, required=False,
                        help="model's name")

    parser.add_argument('--task_type', type=str, required=False,
                        help="the type of the task")

    parser.add_argument('--corpus_type', type=str, required=False,
                        help="the type of corpus")

    parser.add_argument('--model_name', type=str, required=False,
                        help="model's name")

    parser.add_argument('--path_preComputed', type=str, required=False,
                        help="path of training and test set")

    parser.add_argument('--path_aspect_conf', type=str, required=False,
                        help="conf file for evaluation aspect")

    parser.add_argument('--path_json_input', type=str, required=False,
                        help="json template")

    parser.add_argument('--fn_write_json', type=str, required=False,
                        help="the type of the task")

    args = parser.parse_args()

    # Configuration
    path_text = args.text
    path_true_file = args.true_file
    path_pred_file = args.pred_file

    task_type = args.task_type
    corpus_type = args.corpus_type
    model_name = args.model_name

    path_preComputed = args.path_preComputed
    path_aspect_conf = args.path_aspect_conf

    path_json_input = args.path_json_input
    fn_write_json = args.fn_write_json

    # Initalization
    dict_aspect_func = loadConf(path_aspect_conf)
    metric_names = list(dict_aspect_func.keys())
    print("dict_aspect_func: ", dict_aspect_func)
    print(dict_aspect_func)

    fwrite_json = open(fn_write_json, 'w')
    path_comb_output = model_name + "/" + path_text.split("/")[-1]
    # get preComputed paths from conf file
    dict_preComputed_path = {}
    for aspect, func in dict_aspect_func.items():
        is_preComputed = func[2].lower()
        if is_preComputed == "yes":
            dict_preComputed_path[aspect] = path_preComputed + "_" + aspect + ".pkl"
            print("PreComputed directory:\t", dict_preComputed_path[aspect])



    list_text_sent, list_text_token = read_single_column(path_text, 0)
    list_true_tags_sent, list_true_tags_token = read_single_column(path_text, 1)
    list_pred_tags_sent, list_pred_tags_token = read_single_column(path_text, 2)




    # holistic_performance = f1_score(list_true_tags_sent, list_pred_tags_sent)
    #
    # holistic_performance = holistic_performance * 100



    # list_true_tags_sent2 = read_tag(path_true_file)
    # list_pred_tags_sent2 = read_tag(path_pred_file)

    # holistic_performance = f1(list_true_tags_sent, list_pred_tags_sent)["f1"]

    # holistic_performance2 = f1(list_true_tags_sent2, list_pred_tags_sent2)
    # print(holistic_performance)
    # print(holistic_performance2)

    # holistic_performance = f1_score(list_true_tags_token, list_pred_tags_token,average='micro') # 0.8966716343765524
    # print(holistic_performance3)
    # holistic_performance4 = f1_score(list_true_tags_sent2, list_pred_tags_sent2,average='macro')

    # print(holistic_performance2)
    #
    # print(holistic_performance4)
    # exit()
    # print(len(sum(list_text_sent,[])), len(list_text_token))
    # exit()

    # pred_chunks = set(get_chunks(list_true_tags_token[0:20]))
    # print(pred_chunks)

    dict_span2aspectVal, dict_span2sid, dict_chunkid2span = getAspectValue(list_text_token, list_true_tags_token, list_text_sent,
                                                        list_true_tags_sent, dict_preComputed_path, dict_aspect_func)
    dict_span2aspectVal_pred, dict_span2sid_pred, dict_chunkid2span_pred = getAspectValue(list_text_token, list_pred_tags_token, list_text_sent,
                                                                  list_pred_tags_sent, dict_preComputed_path,
                                                                  dict_aspect_func)


    print(len(dict_chunkid2span), len(dict_chunkid2span_pred))

    #holistic_performance = f1(list_true_tags_token, list_pred_tags_token)["f1"]
    holistic_performance = accuracy(list_true_tags_token, list_pred_tags_token)
    #print(holistic_performance, holistic_performance2)
    confidence_low_overall, confidence_up_overall = compute_confidence_interval_f1(dict_span2sid.keys(),
                                                                                   dict_span2sid_pred.keys(),
                                                                                   dict_span2sid, dict_span2sid_pred,
                                                                                   n_times=1000)

    print("------------------ Holistic Result")
    print()
    print(holistic_performance)

    print("confidence_low_overall:\t", confidence_low_overall)
    print("confidence_up_overall:\t", confidence_up_overall)
    # tot = 0
    # for k, v in dict_span2aspectVal.items():
    #     if k == "tLen":
    #         for kk, vv in v.items():
    #             if vv!=1:
    #                 continue
    #             tot+=1
    #             if tot >10:
    #                 break
    #             print(kk, vv)
    # tot = 0
    # for k, v in dict_span2aspectVal_pred.items():
    #     if k == "tLen":
    #         for kk, vv in v.items():
    #             if vv!=1:
    #                 continue
    #             tot+=1
    #             if tot >10:
    #                 break
    #             print(kk, vv)


    # print(dict_span2aspectVal)

    # print(f1(list_true_tags_sent, list_pred_tags_sent))

    # print(f1(list_true_tags_token, list_pred_tags_token)["f1"])

    def __selectBucktingFunc(func_name, func_setting, dict_obj):
        if func_name == "bucketAttribute_SpecifiedBucketInterval":
            return eval(func_name)(dict_obj, eval(func_setting))
        elif func_name == "bucketAttribute_SpecifiedBucketValue":
            if len(func_setting.split("\t")) != 2:
                raise ValueError("selectBucktingFunc Error!")
            n_buckets, specified_bucket_value_list = int(func_setting.split("\t")[0]), eval(func_setting.split("\t")[1])
            return eval(func_name)(dict_obj, n_buckets, specified_bucket_value_list)
        elif func_name == "bucketAttribute_DiscreteValue":  # now the discrete value is R-tag..
            if len(func_setting.split("\t")) != 2:
                raise ValueError("selectBucktingFunc Error!")
            tags_list = list(set(dict_obj.values()))
            topK_buckets, min_buckets = int(func_setting.split("\t")[0]), int(func_setting.split("\t")[1])
            # return eval(func_name)(dict_obj, min_buckets, topK_buckets)
            return eval(func_name)(dict_obj, topK_buckets, min_buckets)


    dict_bucket2span = {}
    dict_bucket2span_pred = {}
    dict_bucket2f1 = {}
    aspect_names = []
    errorCase_list = []

    for aspect, func in dict_aspect_func.items():
        # print(aspect, dict_span2aspectVal[aspect])
        dict_bucket2span[aspect] = __selectBucktingFunc(func[0], func[1], dict_span2aspectVal[aspect])
        # print(aspect, dict_bucket2span[aspect])
        # exit()
        dict_bucket2span_pred[aspect] = bucketAttribute_SpecifiedBucketInterval(dict_span2aspectVal_pred[aspect],
                                                                                dict_bucket2span[aspect].keys())
        dict_bucket2f1[aspect], errorCase_list = getBucketF1_pos(dict_bucket2span[aspect], dict_bucket2span_pred[aspect], dict_span2sid, dict_span2sid_pred, dict_chunkid2span, dict_chunkid2span_pred)
        aspect_names.append(aspect)
    print("aspect_names: ", aspect_names)

    print("------------------ Breakdown Performance")
    for aspect in dict_aspect_func.keys():
        printDict(dict_bucket2f1[aspect], aspect)
    print("")

    # Calculate databias w.r.t numeric attributes
    dict_aspect2bias = {}
    for aspect, aspect2Val in dict_span2aspectVal.items():
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
                                             "confidence_low": confidence_low, "confidence_up": confidence_up, "bucket_error_case":error_entity_list})

    obj_json = load_json(path_json_input)

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

    obj_json["model"]["results"]["overall"]["error_case"] = errorCase_list


    save_json(obj_json, "./instantiate.json")
    save_json(obj_json, fn_write_json)


