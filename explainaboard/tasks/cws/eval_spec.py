import explainaboard.error_analysis as ea


def get_chunks(seq):
	"""
	tags:dic{'per':1,....}
	Args:
		seq: [4, 4, 0, 0, ...] sequence of labels
		tags: dict["O"] = 4
	Returns:
		list of (chunk_type, chunk_start, chunk_end)

	Example:
		seq = [4, 5, 0, 3]
		tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
		result = [("PER", 0, 2), ("LOC", 3, 4)]
	"""
	default = 'O'
	# idx_to_tag = {idx: tag for tag, idx in tags.items()}
	chunks = []
	# chunk_type, chunk_start = None, None
	chunk_current = 0
	#print(seq)
	# BMES -> BIEO
	w_start = 0
	chunk = None
	tag = ""
	for i, tok in enumerate(seq):
		tag += tok
		if tok == "S":
			chunk = ("S",i, i+1)
			chunks.append(chunk)
			tag = ""
		if tok == "B":
			w_start = i
		if tok == "E":
			chunk = (tag, w_start, i+1)
			chunks.append(chunk)
			tag=""




	# for i, tok in enumerate(seq):
	# 	if tok == "M":
	# 		tok = "I"
	# 	if tok == "S":
	# 		tok = "B"
	#
	# 	#End of a chunk 1
	# 	if tok == default and chunk_type is not None:
	# 		# Add a chunk.
	# 		chunk = (chunk_type, chunk_start, i)
	# 		chunks.append(chunk)
	# 		chunk_type, chunk_start = None, None
	#
	# 	# End of a chunk + start of a chunk!
	# 	elif tok != default:
	# 		tok_chunk_class, tok_chunk_type = get_chunk_type(tok)
	# 		if chunk_type is None:
	# 			chunk_type, chunk_start = tok_chunk_type, i
	# 		elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
	# 			chunk = (chunk_type, chunk_start, i)
	# 			chunks.append(chunk)
	# 			chunk_type, chunk_start = tok_chunk_type, i
	# 	else:
	# 		pass
	# # end condition
	# if chunk_type is not None:
	# 	chunk = (chunk_type, chunk_start, len(seq))
	# 	chunks.append(chunk)

	return chunks


def read_data(corpus_type, fn, column_no=-1, delimiter =' '):
	print('corpus_type',corpus_type)
	word_sequences = list()
	tag_sequences = list()
	total_word_sequences = list()
	total_tag_sequences = list()
	with ea.codecs.open(fn, 'r', 'utf-8') as f:
		lines = f.readlines()
	curr_words = list()
	curr_tags = list()
	for k in range(len(lines)):
		line = lines[k].strip()
		if len(line) == 0 or line.startswith('-DOCSTART-'): # new sentence or new document
			if len(curr_words) > 0:
				word_sequences.append(curr_words)
				tag_sequences.append(curr_tags)
				curr_words = list()
				curr_tags = list()
			continue

		strings = line.split(delimiter)
		word = strings[0].strip()
		tag = strings[column_no].strip()  # be default, we take the last tag

		#tag='B-'+tag
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









#   getAspectValue(test_word_sequences, test_trueTag_sequences, test_word_sequences_sent, dict_precomputed_path)

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
	for aspect, fun in dict_aspect_func.items():
		dict_span2aspectVal[aspect] = {}

	eDen_list, sentLen_list = [], []
	eDen_list, sentLen_list = getSententialValue(test_trueTag_sequences_sent,
																	 test_word_sequences_sent)


	dict_pos2sid = ea.getPos2SentId(test_word_sequences_sent)
	dict_ap2rp = ea.getTokenPosition(test_word_sequences_sent)
	all_chunks = ea.get_chunks(test_trueTag_sequences)

	dict_span2sid = {}
	dict_chunkid2span = {}
	for span_info in all_chunks:

		#print(span_info)


		#span_type = span_info[0].lower()

		#print(span_type)
		idx_start = span_info[1]
		idx_end = span_info[2]
		span_cnt = ''.join(test_word_sequences[idx_start:idx_end]).lower()
		#print(span_cnt.encode("utf-8").decode("utf-8"))
		span_cnt = span_cnt.encode("gbk","ignore").decode("gbk","ignore")
		#print(sys.getdefaultencoding())
		span_type = ''.join(test_trueTag_sequences[idx_start:idx_end])


		span_pos = str(idx_start) + "|||" + str(idx_end) + "|||" + span_type

		if len(span_type) !=(idx_end  - idx_start):
			print(idx_start, idx_end)
			print(span_info)
			print(span_type + "\t" + span_cnt)
			print("--------------")


		#print(span_pos)
		# print(span_info)
		# print(span_cnt)


		span_length = idx_end - idx_start

		# span_token_list = test_word_sequences[idx_start:idx_end]
		# span_token_pos_list = [str(pos) + "|||" + span_type for pos in range(idx_start, idx_end)]
		#print(span_token_pos_list)


		span_sentid = dict_pos2sid[idx_start]
		sLen = float(sentLen_list[span_sentid])

		dict_span2sid[span_pos] = span_sentid


		text_sample = "".join(test_word_sequences_sent[span_sentid])
		text_sample = text_sample

		dict_chunkid2span[span_pos] = span_cnt + "|||" + text_sample

		# Sentence Length: sLen
		aspect = "sLen"
		if aspect in dict_aspect_func.keys():
			dict_span2aspectVal[aspect][span_pos] = sLen



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




# def tuple2str(triplet):
# 	res = ""
# 	for v in triplet:
# 		res += str(v) + "_"
# 	return res.rstrip("_")






def evaluate(task_type = "ner", analysis_type = "single", systems = [], output = "./output.json", is_print_ci = False, is_print_case = False, is_print_ece = False):

	path_text = ""

	if analysis_type == "single":
		path_text = systems[0]



	corpus_type = "dataset_name"
	model_name = "model_name"
	path_preComputed = ""
	path_aspect_conf = "./explainaboard/tasks/cws/conf.aspects"
	path_json_input = "./explainaboard/tasks/cws/template.json"
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




	list_text_sent, list_text_token = ea.read_single_column(path_text, 0)
	list_true_tags_sent, list_true_tags_token = ea.read_single_column(path_text, 1)
	list_pred_tags_sent, list_pred_tags_token = ea.read_single_column(path_text, 2)


	dict_span2aspectVal, dict_span2sid, dict_chunkid2span = getAspectValue(list_text_token, list_true_tags_token, list_text_sent, list_true_tags_sent, dict_preComputed_path, dict_aspect_func)
	dict_span2aspectVal_pred, dict_span2sid_pred, dict_chunkid2span_pred  = getAspectValue(list_text_token, list_pred_tags_token, list_text_sent, list_pred_tags_sent, dict_preComputed_path, dict_aspect_func)


	holistic_performance = ea.f1(list_true_tags_sent, list_pred_tags_sent)["f1"]


	confidence_low_overall, confidence_up_overall = 0,0
	if is_print_ci:
		confidence_low_overall, confidence_up_overall = ea.compute_confidence_interval_f1_cws(dict_span2sid.keys(), dict_span2sid_pred.keys(), dict_span2sid, dict_span2sid_pred, n_times=10)



	print("confidence_low_overall:\t", confidence_low_overall)
	print("confidence_up_overall:\t", confidence_up_overall)





	print("------------------ Holistic Result")
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
	errorCase_list = []

	for aspect, func in dict_aspect_func.items():
		# print(aspect, dict_span2aspectVal[aspect])
		dict_bucket2span[aspect] = __selectBucktingFunc(func[0], func[1], dict_span2aspectVal[aspect])
		# print(aspect, dict_bucket2span[aspect])
		# exit()
		dict_bucket2span_pred[aspect] = ea.bucketAttribute_SpecifiedBucketInterval(dict_span2aspectVal_pred[aspect],
																																							 dict_bucket2span[aspect].keys())
		dict_bucket2f1[aspect], errorCase_list = ea.getBucketF1_cws(dict_bucket2span[aspect], dict_bucket2span_pred[aspect], dict_span2sid, dict_span2sid_pred, dict_chunkid2span, dict_chunkid2span_pred, list_true_tags_token, list_pred_tags_token, is_print_ci, is_print_case)
		aspect_names.append(aspect)
	print("aspect_names: ", aspect_names)

	# for v in errorCase_list:
	# 	print(v)




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
			dict_fineGrained[aspect].append({"bucket_name":bucket_name, "bucket_value":bucket_value, "num":n_sample, "confidence_low":confidence_low, "confidence_up":confidence_up, "bucket_error_case":error_entity_list[0:int(len(error_entity_list)/10)]})





	obj_json = ea.load_json(path_json_input)

	obj_json["task"] = task_type
	obj_json["data"]["name"] = corpus_type
	obj_json["data"]["language"] = "Chinese"
	obj_json["data"]["bias"] = dict_aspect2bias

	obj_json["model"]["name"] = model_name
	obj_json["model"]["results"]["overall"]["performance"] = holistic_performance
	obj_json["model"]["results"]["overall"]["confidence_low"] = confidence_low_overall
	obj_json["model"]["results"]["overall"]["confidence_up"] = confidence_up_overall
	obj_json["model"]["results"]["fine_grained"] = dict_fineGrained


	# Save error cases: overall
	obj_json["model"]["results"]["overall"]["error_case"] = errorCase_list[0:int(len(errorCase_list)/10)]



	ea.save_json(obj_json, fn_write_json)

