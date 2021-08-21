# -*- coding: utf-8 -*-
import numpy
import explainaboard.error_analysis as ea
import explainaboard.data_utils as du


def process_all(file_path, size_of_bin=10, dataset='atis', model='lstm-self-attention'):
    """

    :param file_path: the file_path is the path to your file.

    And the path must include file name.

    the file name is in this format: test_dataset_model.tsv.

    the file_path must in the format: /root/path/to/your/file/test_dataset.tsv

    The file must in this format:
    sentence1\tsentence2\tground_truth\tpredict_label\tprobability\tright_or_not
    if prediction is right, right_or_not is assigned to 1, otherwise 0.

    :param size_of_bin: the numbers of how many bins

    :param dataset: the name of the dataset

    :param model: the name of the model

    :return:
    ece :the ece of this file
    dic :the details of the ECE information in json format
    """
    from collections import OrderedDict

    probability_list, right_or_not_list = du.get_probability_right_or_not(file_path, prob_col=4, right_or_not_col=5)

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


def getAspectValue(sent1_list, sent2_list, sample_list_tag, sample_list_tag_pred, dict_aspect_func):
    dict_span2aspectVal = {}
    dict_span2aspectVal_pred = {}

    for aspect, fun in dict_aspect_func.items():
        dict_span2aspectVal[aspect] = {}
        dict_span2aspectVal_pred[aspect] = {}

    # for error analysis
    dict_sid2sentpair = {}

    sample_id = 0
    for sent1, sent2, tag, tag_pred in zip(sent1_list, sent2_list, sample_list_tag, sample_list_tag_pred):

        word_list1 = ea.wordSegment(sent1).split(" ")
        word_list2 = ea.wordSegment(sent2).split(" ")

        # for saving errorlist -- fine-grained version
        dict_sid2sentpair[str(sample_id)] = ea.format4json_tc(
            ea.format4json_tc(sent1) + "|||" + ea.format4json_tc(sent2))

        sent1_length = len(word_list1)
        sent2_length = len(word_list2)

        sent_pos = ea.tuple2str((sample_id, tag))
        sent_pos_pred = ea.tuple2str((sample_id, tag_pred))

        hypo = [ea.wordSegment(sent2)]
        refs = [[ea.wordSegment(sent1)]]

        # bleu = sacrebleu.corpus_bleu(hypo, refs).score * 0.01

        # aspect = "bleu"
        # if aspect in dict_aspect_func.keys():
        # 	dict_span2aspectVal["bleu"][sent_pos] = bleu
        # 	dict_span2aspectVal_pred["bleu"][sent_pos_pred] = bleu

        # Sentence Length: sentALen
        aspect = "sentALen"
        if aspect in dict_aspect_func.keys():
            dict_span2aspectVal[aspect][sent_pos] = float(sent1_length)
            dict_span2aspectVal_pred[aspect][sent_pos_pred] = float(sent1_length)

        # Sentence Length: sentBLen
        aspect = "sentBLen"
        if aspect in dict_aspect_func.keys():
            dict_span2aspectVal["sentBLen"][sent_pos] = float(sent2_length)
            dict_span2aspectVal_pred[aspect][sent_pos_pred] = float(sent2_length)

        # The difference of sentence length: senDeltaLen
        aspect = "A-B"
        if aspect in dict_aspect_func.keys():
            dict_span2aspectVal["A-B"][sent_pos] = float(sent1_length - sent2_length)
            dict_span2aspectVal_pred[aspect][sent_pos_pred] = float(sent1_length - sent2_length)

        # "A+B"
        aspect = "A+B"
        if aspect in dict_aspect_func.keys():
            dict_span2aspectVal["A+B"][sent_pos] = float(sent1_length + sent2_length)
            dict_span2aspectVal_pred[aspect][sent_pos_pred] = float(sent1_length + sent2_length)

        # "A/B"
        aspect = "A/B"
        if aspect in dict_aspect_func.keys():
            dict_span2aspectVal["A/B"][sent_pos] = float(sent1_length * 1.0 / sent2_length)
            dict_span2aspectVal_pred[aspect][sent_pos_pred] = float(sent1_length * 1.0 / sent2_length)

        # Tag: tag
        aspect = "tag"
        if aspect in dict_aspect_func.keys():
            dict_span2aspectVal["tag"][sent_pos] = tag
            dict_span2aspectVal_pred[aspect][sent_pos_pred] = tag

        sample_id += 1
    # print(dict_span2aspectVal["bleu"])
    return dict_span2aspectVal, dict_span2aspectVal_pred, dict_sid2sentpair


def evaluate(task_type="ner", analysis_type="single", systems=[], output="./output.json", is_print_ci=False,
             is_print_case=False, is_print_ece=False):
    path_text = ""

    if analysis_type == "single":
        path_text = systems[0]

    corpus_type = "dataset_name"
    model_name = "model_name"
    path_preComputed = ""
    path_aspect_conf = "./explainaboard/tasks/nli/conf.aspects"
    path_json_input = "./explainaboard/tasks/nli/template.json"
    fn_write_json = output

    # Initalization
    dict_aspect_func = ea.loadConf(path_aspect_conf)
    metric_names = list(dict_aspect_func.keys())
    print("dict_aspect_func: ", dict_aspect_func)
    print(dict_aspect_func)

    fwrite_json = open(fn_write_json, 'w')

    # get preComputed paths from conf file
    dict_preComputed_path = {}
    for aspect, func in dict_aspect_func.items():
        is_preComputed = func[2].lower()
        if is_preComputed == "yes":
            dict_preComputed_path[aspect] = path_preComputed + "_" + aspect + ".pkl"
            print("PreComputed directory:\t", dict_preComputed_path[aspect])

    sent1_list, sent2_list, true_label_list, pred_label_list = ea.file_to_list_nli(path_text)

    errorCase_list = []
    if is_print_case:
        errorCase_list = ea.getErrorCase_nli(sent1_list, sent2_list, true_label_list, pred_label_list)
        print(" -*-*-*- the number of error casse:\t", len(errorCase_list))

    # Confidence Interval of Holistic Performance
    confidence_low, confidence_up = 0, 0
    if is_print_ci:
        confidence_low, confidence_up = ea.compute_confidence_interval_acc(true_label_list, pred_label_list,
                                                                           n_times=100)

    dict_span2aspectVal, dict_span2aspectVal_pred, dict_sid2sentpair = getAspectValue(sent1_list, sent2_list,
                                                                                      true_label_list, pred_label_list,
                                                                                      dict_aspect_func)

    holistic_performance = ea.accuracy(true_label_list, pred_label_list)
    holistic_performance = format(holistic_performance, '.3g')

    print("------------------ Holistic Result----------------------")
    print(holistic_performance)

    # print(f1(list_true_tags_token, list_pred_tags_token)["f1"])

    def __selectBucktingFunc(func_name, func_setting, dict_obj):
        if func_name == "bucketAttribute_SpecifiedBucketInterval":
            return ea.bucketAttribute_SpecifiedBucketInterval(dict_obj, eval(func_setting))
        elif func_name == "bucketAttribute_SpecifiedBucketValue":
            if len(func_setting.split("\t")) != 2:
                raise ValueError("selectBucktingFunc Error!")
            n_buckets, specified_bucket_value_list = int(func_setting.split("\t")[0]), eval(func_setting.split("\t")[1])
            return ea.bucketAttribute_SpecifiedBucketValue(dict_obj, n_buckets, specified_bucket_value_list)
        elif func_name == "bucketAttribute_DiscreteValue":  # now the discrete value is R-tag..
            if len(func_setting.split("\t")) != 2:
                raise ValueError("selectBucktingFunc Error!")
            tags_list = list(set(dict_obj.values()))
            topK_buckets, min_buckets = int(func_setting.split("\t")[0]), int(func_setting.split("\t")[1])
            # return eval(func_name)(dict_obj, min_buckets, topK_buckets)
            return ea.bucketAttribute_DiscreteValue(dict_obj, topK_buckets, min_buckets)
        else:
            raise ValueError(f'Illegal function name {func_name}')

    dict_bucket2span = {}
    dict_bucket2span_pred = {}
    dict_bucket2f1 = {}
    aspect_names = []

    for aspect, func in dict_aspect_func.items():
        # print(aspect, dict_span2aspectVal[aspect])
        dict_bucket2span[aspect] = __selectBucktingFunc(func[0], func[1], dict_span2aspectVal[aspect])
        # print(aspect, dict_bucket2span[aspect])
        # exit()
        dict_bucket2span_pred[aspect] = ea.bucketAttribute_SpecifiedBucketInterval(dict_span2aspectVal_pred[aspect],
                                                                                   dict_bucket2span[aspect].keys())
        # dict_bucket2span_pred[aspect] = __selectBucktingFunc(func[0], func[1], dict_span2aspectVal_pred[aspect])
        dict_bucket2f1[aspect] = ea.getBucketAcc_with_errorCase_nli(dict_bucket2span[aspect],
                                                                    dict_bucket2span_pred[aspect], dict_sid2sentpair,
                                                                    is_print_ci, is_print_case)
        aspect_names.append(aspect)
    print("aspect_names: ", aspect_names)

    print("------------------ Breakdown Performance")
    for aspect in dict_aspect_func.keys():
        ea.printDict(dict_bucket2f1[aspect], aspect)
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
            bucket_value = format(v[0], '.4g')
            n_sample = v[1]
            confidence_low = format(v[2], '.4g')
            confidence_up = format(v[3], '.4g')

            # for saving errorlist -- finegrained version
            bucket_error_case = v[4]

            # instantiation
            dict_fineGrained[aspect].append({"bucket_name": bucket_name, "bucket_value": bucket_value, "num": n_sample,
                                             "confidence_low": confidence_low, "confidence_up": confidence_up,
                                             "bucket_error_case": bucket_error_case})

    obj_json = ea.load_json(path_json_input)

    obj_json["task"] = task_type
    obj_json["data"]["name"] = corpus_type
    obj_json["data"]["language"] = "English"
    obj_json["data"]["bias"] = dict_aspect2bias

    obj_json["model"]["name"] = model_name
    obj_json["model"]["results"]["overall"]["performance"] = holistic_performance
    obj_json["model"]["results"]["overall"]["confidence_low"] = confidence_low
    obj_json["model"]["results"]["overall"]["confidence_up"] = confidence_up
    obj_json["model"]["results"]["fine_grained"] = dict_fineGrained

    # add errorAnalysis -- holistic
    obj_json["model"]["results"]["overall"]["error_case"] = errorCase_list

    # for Calibration
    ece = 0
    dic_calibration = None
    if is_print_ece:
        ece, dic_calibration = process_all(path_text,
                                           size_of_bin=10, dataset=corpus_type, model=model_name)

    obj_json["model"]["results"]["calibration"] = dic_calibration

    ea.save_json(obj_json, fn_write_json)
