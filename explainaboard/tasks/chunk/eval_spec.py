# -*- coding: utf-8 -*-
import explainaboard.error_analysis as ea



def getAspectValue(test_word_sequences, test_trueTag_sequences, test_word_sequences_sent,
				   test_trueTag_sequences_sent, dict_preComputed_path, dict_aspect_func):


	def getSententialValue(test_trueTag_sequences_sent, test_word_sequences_sent):

		eDen = []
		sentLen = []

		for i, test_sent in enumerate(test_trueTag_sequences_sent):
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



	dict_preComputed_model = {}
	for aspect, path in dict_preComputed_path.items():
		print("path:\t"+path)
		if ea.os.path.exists(path):
			print('load the hard dictionary of entity span in test set...')
			fread = open(path, 'rb')
			dict_preComputed_model[aspect] = ea.pickle.load(fread)
		else:
			raise ValueError("can not load hard dictionary" + aspect + "\t" + path)




	dict_span2aspectVal = {}
	dict_chunkid2span = {}
	for aspect, fun in dict_aspect_func.items():
		dict_span2aspectVal[aspect] = {}

	eDen_list, sentLen_list = [], []
	eDen_list, sentLen_list = getSententialValue(test_trueTag_sequences_sent,
																	 test_word_sequences_sent)


	dict_pos2sid = ea.getPos2SentId(test_word_sequences_sent)
	dict_ap2rp = ea.getTokenPosition(test_word_sequences_sent)
	all_chunks = ea.get_chunks(test_trueTag_sequences)
	dict_span2sid = {}
	for span_info in all_chunks:

		span_type = span_info[0].lower()
		#print(span_type)
		idx_start = span_info[1]
		idx_end = span_info[2]
		span_cnt = ' '.join(test_word_sequences[idx_start:idx_end]).lower()
		span_pos = str(idx_start) + "_" + str(idx_end) + "_" + span_type

		span_length = idx_end - idx_start

		span_token_list = test_word_sequences[idx_start:idx_end]
		span_token_pos_list = [ str(pos) + "_" + span_type for pos in range(idx_start, idx_end)]


		span_sentid = dict_pos2sid[idx_start]




		sLen = float(sentLen_list[span_sentid])

		dict_span2sid[span_pos] = span_sentid
		dict_chunkid2span[span_pos] = ea.format4json(span_cnt) + "|||" + ea.format4json(' '.join(test_word_sequences_sent[span_sentid]))


		# Sentence Length: sLen
		aspect = "sLen"
		if aspect in dict_aspect_func.keys():
			dict_span2aspectVal[aspect][span_pos] = sLen


		# Relative Position: relPos
		aspect = "rPos"
		if aspect in dict_aspect_func.keys():
			dict_span2aspectVal[aspect][span_pos] = (dict_ap2rp[idx_start])*1.0/sLen


		# Entity Length: eLen
		aspect = "eLen"
		if aspect in dict_aspect_func.keys():
			dict_span2aspectVal[aspect][span_pos] = float(span_length)


		# Tag: tag
		aspect = "tag"
		if aspect in dict_aspect_func.keys():
			dict_span2aspectVal[aspect][span_pos] = span_type

		#print(dict_span2aspectVal)
	return  dict_span2aspectVal, dict_span2sid, dict_chunkid2span



def tuple2str(triplet):
	res = ""
	for v in triplet:
		res += str(v) + "_"
	return res.rstrip("_")





def evaluate(task_type = "ner", analysis_type = "single", systems = [], output = "./output.json", is_print_ci = False, is_print_case = False, is_print_ece = False):

	path_text = ""

	if analysis_type == "single":
		path_text = systems[0]



	corpus_type = "dataset_name"
	model_name = "model_name"
	path_preComputed = ""
	path_aspect_conf = "./explainaboard/tasks/chunk/conf.aspects"
	path_json_input = "./explainaboard/tasks/chunk/template.json"
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




	list_text_sent, list_text_token = ea.read_single_column(path_text, 0)
	list_true_tags_sent, list_true_tags_token = ea.read_single_column(path_text, 1)
	list_pred_tags_sent, list_pred_tags_token = ea.read_single_column(path_text, 2)



	dict_span2aspectVal, dict_span2sid, dict_chunkid2span  = getAspectValue(list_text_token, list_true_tags_token, list_text_sent, list_true_tags_sent, dict_preComputed_path, dict_aspect_func)
	dict_span2aspectVal_pred, dict_span2sid_pred, dict_chunkid2span_pred = getAspectValue(list_text_token, list_pred_tags_token, list_text_sent, list_pred_tags_sent, dict_preComputed_path, dict_aspect_func)


	holistic_performance = ea.f1(list_true_tags_sent, list_pred_tags_sent)["f1"]
	confidence_low_overall, confidence_up_overall = 0,0
	if is_print_ci:
		confidence_low_overall, confidence_up_overall = ea.compute_confidence_interval_f1(dict_span2sid.keys(), dict_span2sid_pred.keys(), dict_span2sid, dict_span2sid_pred, n_times=1000)

	# print(dict_span2aspectVal)

	print("confidence_low_overall:\t", confidence_low_overall)
	print("confidence_up_overall:\t", confidence_up_overall)
	# holistic_performance = f1(list_true_tags_sent, list_pred_tags_sent)["f1"]
	#print(f1(list_true_tags_sent, list_pred_tags_sent))





	print("------------------ Holistic Result")
	print(holistic_performance)



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
			raise ValueError(f'Illegal bucketing function {func_name}')


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
		dict_bucket2span_pred[aspect] = ea.bucketAttribute_SpecifiedBucketInterval(dict_span2aspectVal_pred[aspect],
																																							 dict_bucket2span[aspect].keys())
		dict_bucket2f1[aspect], errorCase_list = ea.getBucketF1_chunk(dict_bucket2span[aspect], dict_bucket2span_pred[aspect], dict_span2sid, dict_span2sid_pred, dict_chunkid2span, dict_chunkid2span_pred, is_print_ci, is_print_case)
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
			bucket_value   = format(float(v[0])*100, '.4g')
			n_sample = v[1]
			confidence_low = format(float(v[2])*100, '.4g')
			confidence_up  = format(float(v[3])*100, '.4g')
			error_entity_list = v[4]
			# instantiation
			dict_fineGrained[aspect].append({"bucket_name":bucket_name, "bucket_value":bucket_value, "num":n_sample, "confidence_low":confidence_low, "confidence_up":confidence_up, "bucket_error_case":error_entity_list})





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


	obj_json["model"]["results"]["overall"]["error_case"] = errorCase_list


	ea.save_json(obj_json, "./instantiate.json")
	ea.save_json(obj_json, fn_write_json)

