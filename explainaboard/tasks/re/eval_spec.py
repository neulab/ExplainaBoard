# -*- coding: utf-8 -*-
import explainaboard.error_analysis as ea


def sent2list(sent):
	if len(sent.split(" ")) == 1 and len(list(sent))>=5:
		return list(sent)
	else:
		return sent.split(" ")




def getAspectValue(sample_list, dict_aspect_func):


	dict_span2aspectVal = {}
	dict_span2aspectVal_pred = {}


	for aspect, fun in dict_aspect_func.items():
		dict_span2aspectVal[aspect] = {}
		dict_span2aspectVal_pred[aspect] = {}


	# maintain it for print error case
	dict_sid2sent = {}


	sample_id = 0
	for  info_list in sample_list:



		#
		#
		#
		# word_list = wordSegment(sent).split(" ")

		# Sentence	Entities	Paragraph	True Relation Label	Predicted Relation Label
		# Sentence Length	Paragraph Length	Number of Entities in Ground Truth Relation	Average Distance of Entities

		sent, entities, paragraph, true_label, pred_label, sent_length, para_length, n_entity, avg_distance = info_list

		dict_sid2sent[str(sample_id)] = ea.format4json_tc(entities + "|||" + sent)

		sent_pos = ea.tuple2str((sample_id, true_label))
		sent_pos_pred = ea.tuple2str((sample_id, pred_label))

		# Sentence Length: sentALen
		aspect = "sLen"
		if aspect in dict_aspect_func.keys():
			dict_span2aspectVal[aspect][sent_pos] = float(sent_length)
			dict_span2aspectVal_pred[aspect][sent_pos_pred] = float(sent_length)

		# Paragraph Length: pLen
		aspect = "pLen"
		if aspect in dict_aspect_func.keys():
			dict_span2aspectVal[aspect][sent_pos] = float(para_length)
			dict_span2aspectVal_pred[aspect][sent_pos_pred] = float(para_length)

		# Number of Entity: nEnt
		aspect = "nEnt"
		if aspect in dict_aspect_func.keys():
			dict_span2aspectVal[aspect][sent_pos] = float(n_entity)
			dict_span2aspectVal_pred[aspect][sent_pos_pred] = float(n_entity)

		# Average Distance: avgDist
		aspect = "avgDist"
		if aspect in dict_aspect_func.keys():
			dict_span2aspectVal[aspect][sent_pos] = float(avg_distance)
			dict_span2aspectVal_pred[aspect][sent_pos_pred] = float(avg_distance)



		# Tag: tag
		aspect = "tag"   ############## MUST Be Gold Tag for text classification task
		if aspect in dict_aspect_func.keys():
			dict_span2aspectVal[aspect][sent_pos] = true_label
			dict_span2aspectVal_pred[aspect][sent_pos_pred] = true_label


		sample_id += 1

	# print(dict_span2aspectVal["bleu"])
	return  dict_span2aspectVal, dict_span2aspectVal_pred, dict_sid2sent



def evaluate(task_type = "ner", analysis_type = "single", systems = [], output = "./output.json", is_print_ci = False, is_print_case = False, is_print_ece = False):

	path_text = ""

	if analysis_type == "single":
		path_text = systems[0]



	corpus_type = "dataset_name"
	model_name = "model_name"
	path_preComputed = ""
	path_aspect_conf = "./explainaboard/tasks/re/conf.aspects"
	path_json_input = "./explainaboard/tasks/re/template.json"
	# path_aspect_conf = "./tasks/re/conf.aspects"
	# path_json_input = "./tasks/re/template.json"
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


	sample_list, sent_list, entity_list, true_list, pred_list = ea.file_to_list_re(path_text)

	errorCase_list = []
	if is_print_case:
		errorCase_list = ea.getErrorCase_re(sent_list, entity_list, true_list, pred_list)
		print(" -*-*-*- the number of error casse:\t", len(errorCase_list))




	dict_span2aspectVal, dict_span2aspectVal_pred,  dict_sid2sent  = getAspectValue(sample_list, dict_aspect_func)


	holistic_performance = ea.accuracy(true_list, pred_list)
	holistic_performance = format(holistic_performance, '.3g')


	# Confidence Interval of Holistic Performance
	confidence_low, confidence_up = 0,0
	if is_print_ci:
		confidence_low, confidence_up = ea.compute_confidence_interval_acc(true_list, pred_list, n_times=1000)



	dict_span2aspectVal, dict_span2aspectVal_pred,  dict_sid2sent  = getAspectValue(sample_list, dict_aspect_func)





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
		dict_bucket2span_pred[aspect] = ea.bucketAttribute_SpecifiedBucketInterval(dict_span2aspectVal_pred[aspect],
																																							 dict_bucket2span[aspect].keys())
		# dict_bucket2span_pred[aspect] = __selectBucktingFunc(func[0], func[1], dict_span2aspectVal_pred[aspect])
		dict_bucket2f1[aspect] = ea.getBucketAcc_with_errorCase_re(dict_bucket2span[aspect], dict_bucket2span_pred[aspect], dict_sid2sent, is_print_ci, is_print_case)
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
			confidence_low_bucket = format(v[2], '.4g')
			confidence_up_bucket  = format(v[3], '.4g')
			bucket_error_case = v[4]

			# instantiation
			dict_fineGrained[aspect].append({"bucket_name":bucket_name, "bucket_value":bucket_value, "num":n_sample, "confidence_low":confidence_low_bucket, "confidence_up":confidence_up_bucket, "bucket_error_case":bucket_error_case})





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



	ea.save_json(obj_json, fn_write_json)












#
# def main():
#
# 	parser = argparse.ArgumentParser(description='Interpretable Evaluation for NLP')
#
#
# 	parser.add_argument('--task', type=str, required=True,
# 						help="absa")
#
# 	parser.add_argument('--ci', type=str, required=False, default= False,
# 						help="True|False")
#
# 	parser.add_argument('--case', type=str, required=False, default= False,
# 						help="True|False")
#
# 	parser.add_argument('--ece', type=str, required=False, default= False,
# 						help="True|False")
#
#
# 	parser.add_argument('--type', type=str, required=False, default="single",
# 						help="analysis type: single|pair|combine")
# 	parser.add_argument('--systems', type=str, required=True,
# 						help="the directories of system outputs. Multiple one should be separated by comma, for example, system1,system2 (no space)")
#
# 	parser.add_argument('--output', type=str, required=True,
# 						help="analysis output file")
# 	args = parser.parse_args()
#
#
# 	is_print_ci = args.ci
# 	is_print_case = args.case
# 	is_print_ece = args.ece
#
# 	task = args.task
# 	analysis_type = args.type
# 	systems = args.systems.split(",")
# 	output = args.output
#
#
# 	print("task", task)
# 	print("type", analysis_type)
# 	print("systems", systems)
# 	# sample_list = file_to_list_re(systems[0])
# 	# print(sample_list[0])
# 	evaluate(task_type=task, analysis_type=analysis_type, systems=systems, output=output, is_print_ci = is_print_ci, is_print_case = is_print_case, is_print_ece = is_print_ece)
#
# # python eval_spec.py  --task re --systems ./test_re.tsv --output ./a.json
# if __name__ == '__main__':
# 	main()