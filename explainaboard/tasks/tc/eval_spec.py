# -*- coding: utf-8 -*-
import explainaboard.error_analysis as ea

def sent2list(sent):
	if len(sent.split(" ")) == 1 and len(list(sent))>=5:
		return list(sent)
	else:
		return sent.split(" ")




def getAspectValue(sent_list, sample_list_tag, sample_list_tag_pred, dict_aspect_func):
	dict_span2aspectVal = {}
	dict_span2aspectVal_pred = {}


	for aspect, fun in dict_aspect_func.items():
		dict_span2aspectVal[aspect] = {}
		dict_span2aspectVal_pred[aspect] = {}


	# maintain it for print error case
	dict_sid2sent = {}


	sample_id = 0
	for  sent, tag, tag_pred in zip(sent_list, sample_list_tag, sample_list_tag_pred):


		dict_sid2sent[str(sample_id)] = ea.format4json_tc(sent)



		word_list = ea.wordSegment(sent).split(" ")



		sent_length = len(word_list)

		sent_pos = ea.tuple2str((sample_id, tag))
		sent_pos_pred = ea.tuple2str((sample_id, tag_pred))

		# Sentence Length: sentALen
		aspect = "sLen"
		if aspect in dict_aspect_func.keys():
			dict_span2aspectVal[aspect][sent_pos] = float(sent_length)
			dict_span2aspectVal_pred[aspect][sent_pos_pred] = float(sent_length)


		# Tag: tag
		aspect = "tag"   ############## MUST Be Gold Tag for text classification task
		if aspect in dict_aspect_func.keys():
			dict_span2aspectVal[aspect][sent_pos] = tag
			dict_span2aspectVal_pred[aspect][sent_pos_pred] = tag



		sample_id += 1

	# print(dict_span2aspectVal["bleu"])
	return  dict_span2aspectVal, dict_span2aspectVal_pred, dict_sid2sent


def get_probability_right_or_not(file_path):
    """

    :param file_path: the file_path is the path to your file.

    And the path must include file name.

    the file name is in this format: test_dataset_model.tsv.

    the file_path must in the format: /root/path/to/your/file/test_dataset.tsv

    The file must in this format:
    sentence\tground_truth\tpredict_label\tprobability\tright_or_not
    if prediction is right, right_or_not is assigned to 1, otherwise 0.

    """

    import pandas as pd
    import numpy as np

    result = pd.read_csv(file_path, sep='\t', header=None)

    probability_list = np.array(result[3]).tolist()
    right_or_not_list = np.array(result[4]).tolist()

    return probability_list, right_or_not_list


def get_raw_list(probability_list, right_or_not_list):
    total_raw_list = []

    for index in range(len(right_or_not_list)):
        total_raw_list.append([probability_list[index], right_or_not_list[index]])
    return total_raw_list


def calculate_ece(result_list):
    ece = 0
    size = 0
    tem_list = []
    for value in result_list:
        if value[2] == 0:
            tem_list.append(0)
            continue
        size = size + value[2]
        error = abs(float(value[0]) - float(value[1]))
        tem_list.append(error)

    if size == 0:
        return -1

    for i in range(len(result_list)):
        ece = ece + ((result_list[i][2] / size) * tem_list[i])

    return ece


def divide_into_bin(size_of_bin, raw_list):
    bin_list = []
    basic_width = 1 / size_of_bin

    for i in range(0, size_of_bin):
        bin_list.append([])

    for value in raw_list:
        probability = value[0]
        isRight = value[1]
        if probability == 1.0:
            bin_list[size_of_bin - 1].append([probability, isRight])
            continue
        for i in range(0, size_of_bin):
            if (probability >= i * basic_width) & (probability < (i + 1) * basic_width):
                bin_list[i].append([probability, isRight])

    result_list = []
    for i in range(0, size_of_bin):
        value = bin_list[i]
        if len(value) == 0:
            result_list.append([0, 0, 0])
            continue
        total_probability = 0
        total_right = 0
        for result in value:
            total_probability = total_probability + result[0]
            total_right = total_right + result[1]
        result_list.append([total_probability / len(value), total_right / (len(value)), len(value)])

    return result_list


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

    probability_list, right_or_not_list = get_probability_right_or_not(file_path)

    raw_list = get_raw_list(probability_list, right_or_not_list)

    bin_list = divide_into_bin(size_of_bin, raw_list)

    ece = calculate_ece(bin_list)
    dic = ea.OrderedDict()
    dic['dataset-name'] = dataset
    dic['model-name'] = model
    dic['ECE'] = ece
    dic['details'] = []
    basic_width = 1 / size_of_bin
    for i in range(len(bin_list)):
        tem_dic = {}

        bin_name = format(i * basic_width, '.2g') + '-' + format((i+1) * basic_width, '.2g')
        tem_dic = {'interval':bin_name,'average_accuracy': bin_list[i][1], 'average_confidence': bin_list[i][0],
                             'samples_number_in_this_bin': bin_list[i][2]}
        dic['details'].append(tem_dic)

    return ece, dic

def evaluate(task_type = "ner", analysis_type = "single", systems = [], output = "./output.json", is_print_ci = False, is_print_case = False, is_print_ece = False):

	path_text = ""

	if analysis_type == "single":
		path_text = systems[0]



	corpus_type = "dataset_name"
	model_name = "model_name"
	path_preComputed = ""
	path_aspect_conf = "./explainaboard/tasks/tc/conf.aspects"
	path_json_input = "./explainaboard/tasks/tc/template.json"
	fn_write_json = output


	# Initalization
	dict_aspect_func = ea.loadConf(path_aspect_conf)
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



	sent_list, true_label_list, pred_label_list = ea.file_to_list_tc(path_text)

	errorCase_list = []
	if is_print_case:
		errorCase_list = ea.getErrorCase_tc(sent_list, true_label_list, pred_label_list)
		print(" -*-*-*- the number of error casse:\t", len(errorCase_list))












	# Confidence Interval of Holistic Performance
	confidence_low, confidence_up = 0,0
	if is_print_ci:
		confidence_low, confidence_up = ea.compute_confidence_interval_acc(true_label_list, pred_label_list, n_times=1000)




	dict_span2aspectVal, dict_span2aspectVal_pred,  dict_sid2sent  = getAspectValue(sent_list, true_label_list, pred_label_list, dict_aspect_func)


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
			raise ValueError(f'Illegal function type {func_name}')


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
		dict_bucket2f1[aspect] = ea.getBucketAcc_with_errorCase(dict_bucket2span[aspect], dict_bucket2span_pred[aspect], dict_sid2sent, is_print_ci, is_print_case)
		aspect_names.append(aspect)
	print("aspect_names: ", aspect_names)




	print("------------------ Breakdown Performance")
	for aspect in dict_aspect_func.keys():
		ea.printDict(dict_bucket2f1[aspect], aspect)
	print("")


	# Calculate databias w.r.t numeric attributes
	dict_aspect2bias={}
	for aspect, aspect2Val in dict_span2aspectVal.items():
		if type(list(aspect2Val.values())[0]) != type("string"):
			dict_aspect2bias[aspect] = ea.numpy.average(list(aspect2Val.values()))

	print("------------------ Dataset Bias")
	for k,v in dict_aspect2bias.items():
		print(k+":\t"+str(v))
	print("")





	def beautifyInterval(interval):

		if type(interval[0]) == type("string"): ### pay attention to it
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

			#bucket_value = format(v[0]*100,'.4g')
			bucket_value   = format(v[0], '.4g')
			n_sample = v[1]
			confidence_low = format(v[2], '.4g')
			confidence_up  = format(v[3], '.4g')
			bucket_error_case = v[4]

			# instantiation
			dict_fineGrained[aspect].append({"bucket_name":bucket_name, "bucket_value":bucket_value, "num":n_sample, "confidence_low":confidence_low, "confidence_up":confidence_up, "bucket_error_case":bucket_error_case})





	obj_json = ea.load_json(path_json_input)

	obj_json["task"] = task_type
	obj_json["data"]["name"] = corpus_type
	obj_json["data"]["language"] = "English"
	obj_json["data"]["bias"] = dict_aspect2bias
	obj_json["data"]["output"] = path_comb_output

	obj_json["model"]["name"] = model_name
	obj_json["model"]["results"]["overall"]["error_case"] = errorCase_list
	obj_json["model"]["results"]["overall"]["performance"] = holistic_performance
	obj_json["model"]["results"]["overall"]["confidence_low"] = confidence_low
	obj_json["model"]["results"]["overall"]["confidence_up"] = confidence_up
	obj_json["model"]["results"]["fine_grained"] = dict_fineGrained


	ece = 0
	dic_calibration = None
	if is_print_ece:
		ece, dic_calibration = process_all(path_text,
							   size_of_bin=10, dataset=corpus_type, model=model_name)

	obj_json["model"]["results"]["calibration"] = dic_calibration
	#print(dic_calibration)


	ea.save_json(obj_json, "./instantiate.json")
	ea.save_json(obj_json, fn_write_json)


