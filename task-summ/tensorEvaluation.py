# -*- coding: utf-8 -*-
import argparse
import numpy
import sys
sys.path.append("../src")
from utils import *
import sacrebleu
from rouge_metric import PyRouge






def tuple2str(triplet):
	res = ""
	for v in triplet:
		res += str(v) + "_"
	return res.rstrip("_")




def sent2list(sent):
	if len(sent.split(" ")) == 1 and len(list(sent))>=5:
		return list(sent)
	else:
		return sent.split(" ")




def getAspectValue(doc_list, hyp_list, ref_list, r1_list, r2_list, rl_list, dict_aspect_func):



	dict_span2aspectVal = {}

	for aspect, fun in dict_aspect_func.items():
		dict_span2aspectVal[aspect] = {}

	rouge = PyRouge(rouge_n=(1, 2, 4), rouge_l=True, rouge_w=True,
					rouge_w_weight=1.2, rouge_s=False, rouge_su=False, skip_gap=4)

	dict_id2r1={}
	dict_id2r2 = {}
	dict_id2rl = {}


	sample_id = 0
	for  doc, hyp, ref in zip(doc_list, hyp_list, ref_list):

		dict_id2r1[sample_id] = float(r1_list[sample_id])
		dict_id2r2[sample_id] = float(r2_list[sample_id])
		dict_id2rl[sample_id] = float(rl_list[sample_id])

		#density, coverage, compression, copy_len, novelty1, novelty2, repetition1, repetition2 = distance(doc, ref)

		compression = len(doc.split(" "))/len(ref.split(" "))


		# compression = 0.5

		# print(compression)


		# scores = rouge.evaluate([doc], [[ref]])
		# r1_dr = scores["rouge-1"]["f"] * 0.01
		# r2_dr = scores["rouge-2"]["f"] * 0.01
		# r4_dr = scores["rouge-4"]["f"]
		# rL_dr = scores["rouge-l"]["f"]


		r1_dr = 0
		r2_dr = 0
		r4_dr = 0
		rl_dr = 0

		bleu_dr = 0


		# bleu_dr = sacrebleu.corpus_bleu([doc], [[ref]]).score * 0.01
		# print(bleu_dr )



		r1 = r1_list[sample_id]
		r2 = r2_list[sample_id]
		rl = rl_list[sample_id]




		sample_pos_r1 = tuple2str((sample_id, r1))
		sample_pos_r2 = tuple2str((sample_id, r2))
		sample_pos_rl = tuple2str((sample_id, rl))

		length_doc = len(doc.split(" "))
		length_hyp = len(hyp.split(" "))
		length_ref = len(ref.split(" "))


		# Sentence Length: sentALen
		# aspect = "density_r1"
		# if aspect in dict_aspect_func.keys():
		# 	dict_span2aspectVal[aspect][sample_pos_r1] = density


		# Sentence Length: sentALen
		aspect = "compress_r1"
		if aspect in dict_aspect_func.keys():
			# print(aspect)
			dict_span2aspectVal[aspect][sample_pos_r1] = compression

		#
		# aspect = "drDis1_r1"
		# if aspect in dict_aspect_func.keys():
		# 	# print(aspect)
		# 	dict_span2aspectVal[aspect][sample_pos_r1] = r1_dr
		#
		# aspect = "drDis2_r1"
		# if aspect in dict_aspect_func.keys():
		# 	# print(aspect)
		# 	dict_span2aspectVal[aspect][sample_pos_r1] = r2_dr
		# # Sentence Length: sentALen
		# aspect = "copylen_r1"
		# if aspect in dict_aspect_func.keys():
		# 	dict_span2aspectVal[aspect][sample_pos_r1] = copy_len
		#
		# # Sentence Length: sentALen
		# aspect = "novelty1_r1"
		# if aspect in dict_aspect_func.keys():
		# 	dict_span2aspectVal[aspect][sample_pos_r1] = novelty1


		# # Sentence Length: sentALen
		# aspect = "novelty2_r1"
		# if aspect in dict_aspect_func.keys():
		# 	dict_span2aspectVal[aspect][sample_pos_r1] = novelty2



		# Sentence Length: sentALen
		# aspect = "repetition1_r1"
		# if aspect in dict_aspect_func.keys():
		# 	dict_span2aspectVal[aspect][sample_pos_r1] = repetition1


		# # Sentence Length: sentALen
		# aspect = "repetition1_r2"
		# if aspect in dict_aspect_func.keys():
		# 	dict_span2aspectVal[aspect][sample_pos_r1] = repetition2



		# Sentence Length: sentALen
		aspect = "docLen_r1"
		if aspect in dict_aspect_func.keys():
			dict_span2aspectVal[aspect][sample_pos_r1] = float(length_doc)

		aspect = "hypLen_r1"
		if aspect in dict_aspect_func.keys():
			dict_span2aspectVal[aspect][sample_pos_r1] = float(length_hyp)

		aspect = "refLen_r1"
		if aspect in dict_aspect_func.keys():
			dict_span2aspectVal[aspect][sample_pos_r1] = float(length_ref)

		# Sentence Length: sentALen
		aspect = "docLen_r2"
		if aspect in dict_aspect_func.keys():
			dict_span2aspectVal[aspect][sample_pos_r2] = float(length_doc)

		aspect = "hypLen_r2"
		if aspect in dict_aspect_func.keys():
			dict_span2aspectVal[aspect][sample_pos_r2] = float(length_hyp)

		aspect = "refLen_r2"
		if aspect in dict_aspect_func.keys():
			dict_span2aspectVal[aspect][sample_pos_r2] = float(length_ref)

		sample_id += 1

	# print(dict_span2aspectVal["bleu"])
	return  dict_span2aspectVal


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

	path_comb_output = model_name + "/" + path_text.split("/")[-1]


	# get preComputed paths from conf file
	dict_preComputed_path = {}
	for aspect, func in dict_aspect_func.items():
		is_preComputed = func[2].lower()
		if is_preComputed == "yes":
			dict_preComputed_path[aspect] = path_preComputed + "_" + aspect + ".pkl"
			print("PreComputed directory:\t", dict_preComputed_path[aspect])

	doc_list, hyp_list, ref_list, r1, r2, rl, r1_overall, r2_overall, rl_overall = file_to_list_summ(path_text)

	# print(doc_list)
	#
	# exit()



	dict_span2aspectVal= getAspectValue(doc_list, hyp_list, ref_list, r1, r2, rl, dict_aspect_func)


	#holistic_performance = (float(r1_overall[0]) + float(r2_overall[0]) + float(rl_overall[0])) /3
	holistic_performance = {'R1':format(float(r1_overall[0])*100, '.4g'), 'R2':format(float(r2_overall[0])*100, '.4g'), 'RL':format(float(rl_overall[0])*100, '.4g')}




	print("------------------ Holistic Result")
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

		dict_bucket2f1[aspect] = getBucketROUGE(dict_bucket2span[aspect])
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

			bucket_value = format(v[0]*100,'.4g')

			n_sample = v[1]
			confidence = 0
			# instantiation
			dict_fineGrained[aspect].append({"bucket_name":bucket_name, "bucket_value":bucket_value, "num":n_sample, "confidence":confidence})





	obj_json = load_json(path_json_input)

	obj_json["task"] = task_type
	obj_json["data"]["name"] = corpus_type
	obj_json["data"]["output"] = path_comb_output
	obj_json["data"]["language"] = "English"
	obj_json["data"]["bias"] = dict_aspect2bias

	obj_json["model"]["name"] = model_name
	obj_json["model"]["results"]["overall"]["performance"] = holistic_performance
	obj_json["model"]["results"]["fine_grained"] = dict_fineGrained

	save_json(obj_json, "./instantiate.json")
	save_json(obj_json, fn_write_json)


