# -*- coding: utf-8 -*-
import argparse
import numpy
import sys
import csv
sys.path.append("../src")
from errorAnalysis import *
from utils import *
import sacrebleu
from rouge_metric import PyRouge










def sent2list(sent):
	if len(sent.split(" ")) == 1 and len(list(sent))>=5:
		return list(sent)
	else:
		return sent.split(" ")




def getAspectValue(info_list, dict_aspect_func):


	def _convert(string_info):
		if string_info == "n/a":
			return float(0)
		else:
			return float(string_info)


	dict_span2aspectVal = {}
	dict_span2aspectVal_pred = {}


	for aspect, fun in dict_aspect_func.items():
		dict_span2aspectVal[aspect] = {}
		dict_span2aspectVal_pred[aspect] = {}



	# maintain it for print error case
	dict_sid2sent = {}



	for sample_id, info in enumerate(info_list):


		tag = "1"
		tag_pred = "1"
		if info[3] != "1":
			tag_pred = "0"

		sent_pos      = tuple2str((sample_id, tag))
		sent_pos_pred = tuple2str((sample_id, tag_pred))

		dict_sid2sent[str(sample_id)] = format4json_tc(info[0]) + "|||" +  format4json_tc(info[1]) + "|||" + format4json_tc(info[2])




		question_len = _convert(info[4])
		gold_query_len = _convert(info[5])
		gold_query_depth = _convert(info[6])
		hardness = info[7]


		# print(question_len)
		# print(gold_query_len)
		# print(gold_query_depth)
		# print(hardness)



		aspect = "srcLen"
		if aspect in dict_aspect_func.keys():
			dict_span2aspectVal[aspect][sent_pos] = question_len
			dict_span2aspectVal_pred[aspect][sent_pos_pred] = question_len


		aspect = "tgtLen"
		if aspect in dict_aspect_func.keys():
			dict_span2aspectVal[aspect][sent_pos] = gold_query_len
			dict_span2aspectVal_pred[aspect][sent_pos_pred] = gold_query_len

		aspect = "tgtDep"
		if aspect in dict_aspect_func.keys():
			dict_span2aspectVal[aspect][sent_pos] = gold_query_depth
			dict_span2aspectVal_pred[aspect][sent_pos_pred] = gold_query_depth

		aspect = "hard"
		if aspect in dict_aspect_func.keys():
			dict_span2aspectVal[aspect][sent_pos] = hardness
			dict_span2aspectVal_pred[aspect][sent_pos_pred] = hardness



	# print(dict_span2aspectVal["bleu"])
	return  dict_span2aspectVal, dict_span2aspectVal_pred, dict_sid2sent

#
# def getAspectValue(sent_list, sample_list_tag, dict_aspect_func):
# 	dict_span2aspectVal = {}
#
# 	for aspect, fun in dict_aspect_func.items():
# 		dict_span2aspectVal[aspect] = {}
#
# 	sample_id = 0
# 	print("sample_id ---debug\t", sample_id)
# 	for sent, tag in zip(sent_list, sample_list_tag):
#
# 		# word_list = wordSegment(sent).split(" ")
#
# 		word_list = sent.split(" ")
# 		sent_length = len(word_list)
#
# 		sent_pos = tuple2str((sample_id, tag))
#
#
# 		# Sentence Length: sentALen
# 		aspect = "sLen"
# 		if aspect in dict_aspect_func.keys():
# 			dict_span2aspectVal[aspect][sent_pos] = tag
#
#
# 		# Tag: tag
# 		aspect = "tag"
# 		if aspect in dict_aspect_func.keys():
# 			dict_span2aspectVal[aspect][sent_pos] = tag
#
#
# 		sample_id += 1
#
# 	# print(dict_span2aspectVal["bleu"])
# 	print("sample_id ---debug\t", sample_id)
# 	return dict_span2aspectVal


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Learning tagger using neural networks')


	parser.add_argument('--text', type=str, required=False,
						help="the type of the task")


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




	# get preComputed paths from conf file
	dict_preComputed_path = {}
	for aspect, func in dict_aspect_func.items():
		is_preComputed = func[2].lower()
		if is_preComputed == "yes":
			dict_preComputed_path[aspect] = path_preComputed + "_" + aspect + ".pkl"
			print("PreComputed directory:\t", dict_preComputed_path[aspect])


	# src_text_list = []
	# gold_query_list = []
	# pred_query_list = []
	# label_list = []
	#



	info_list = []
	true_label_list, pred_label_list = [], []
	text_list = []
	sql_pred_list = []
	sql_true_list = []
	is_match_list = []

	with open(path_text, newline='') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')

		for ind, row in enumerate(spamreader):
			if ind == 0:
				continue

			tag = "1"
			tag_pred = "1"
			if row[3] != "1":
				tag_pred = "0"

			true_label_list.append(tag)
			pred_label_list.append(tag_pred)
			info_list.append(row)

			text_list.append(row[0])
			sql_true_list.append(row[1])
			sql_pred_list.append(row[2])
			is_match_list.append(row[3])


			#print(row)
			#print(', '.join(row))



	errorCase_list = getErrorCase_semp(text_list, sql_true_list, sql_pred_list, is_match_list)

	#sent1_list, sent2_list, true_label_list, pred_label_list = file_to_list_nli(path_text)


	# print(len(true_label_list))
	# print(len(pred_label_list))
	# exit()


	# Confidence Interval of Holistic Performance
	confidence_low, confidence_up = compute_confidence_interval_acc(true_label_list, pred_label_list, n_times=100)

	# confidence_low, confidence_up = 0,0


	dict_span2aspectVal, dict_span2aspectVal_pred, dict_sid2sent  = getAspectValue(info_list, dict_aspect_func)


	#holistic_performance = accuracy(true_label_list, pred_label_list)["accuracy"]
	holistic_performance = accuracy(true_label_list, pred_label_list)
	holistic_performance = format(holistic_performance, '.3g')





	print("------------------ Holistic Result----------------------")
	print(holistic_performance)
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

	for aspect, func in dict_aspect_func.items():
		# print(aspect, dict_span2aspectVal[aspect])
		dict_bucket2span[aspect] = __selectBucktingFunc(func[0], func[1], dict_span2aspectVal[aspect])
		# print(aspect, dict_bucket2span[aspect])
		# exit()
		dict_bucket2span_pred[aspect] = bucketAttribute_SpecifiedBucketInterval(dict_span2aspectVal_pred[aspect],
																				dict_bucket2span[aspect].keys())
		# dict_bucket2span_pred[aspect] = __selectBucktingFunc(func[0], func[1], dict_span2aspectVal_pred[aspect])
		# dict_bucket2f1[aspect] = getBucketAcc(dict_bucket2span[aspect], dict_bucket2span_pred[aspect])
		# aspect_names.append(aspect)

		dict_bucket2f1[aspect] = getBucketAcc_with_errorCase_semp(dict_bucket2span[aspect], dict_bucket2span_pred[aspect], dict_sid2sent)
		aspect_names.append(aspect)

	print("aspect_names: ", aspect_names)




	print("------------------ Breakdown Performance")
	for aspect in dict_aspect_func.keys():
		printDict(dict_bucket2f1[aspect], aspect)
	print("")


	# Calculate databias w.r.t numeric attributes
	dict_aspect2bias={}
	for aspect, aspect2Val in dict_span2aspectVal.items():
		if type(list(aspect2Val.values())[0]) != type("string"):
			dict_aspect2bias[aspect] = numpy.average(list(aspect2Val.values()))

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





	obj_json = load_json(path_json_input)

	obj_json["task"] = task_type
	obj_json["data"]["name"] = corpus_type
	obj_json["data"]["language"] = "English"
	obj_json["data"]["bias"] = dict_aspect2bias

	obj_json["model"]["name"] = model_name
	obj_json["model"]["results"]["overall"]["performance"] = holistic_performance
	obj_json["model"]["results"]["overall"]["confidence_low"] = confidence_low
	obj_json["model"]["results"]["overall"]["confidence_up"] = confidence_up
	obj_json["model"]["results"]["fine_grained"] = dict_fineGrained

	obj_json["model"]["results"]["overall"]["error_case"] = errorCase_list

	save_json(obj_json, "./instantiate.json")
	save_json(obj_json, fn_write_json)


