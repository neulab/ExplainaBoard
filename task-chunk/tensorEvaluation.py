# -*- coding: utf-8 -*-
import argparse
import numpy
import sys
sys.path.append("../src")
from errorAnalysis import *
from utils import *


def get_chunk_type(tok):
	"""
	Args:
		tok: id of token, ex 4
		idx_to_tag: dictionary {4: "B-PER", ...}
	Returns:
		tuple: "B", "PER"
	"""
	# tag_name = idx_to_tag[tok]
	tag_class = tok.split('-')[0]
	tag_type = tok.split('-')[-1]
	return tag_class, tag_type

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
	chunk_type, chunk_start = None, None
	for i, tok in enumerate(seq):
		#End of a chunk 1
		if tok == default and chunk_type is not None:
			# Add a chunk.
			chunk = (chunk_type, chunk_start, i)
			chunks.append(chunk)
			chunk_type, chunk_start = None, None

		# End of a chunk + start of a chunk!
		elif tok != default:
			tok_chunk_class, tok_chunk_type = get_chunk_type(tok)
			if chunk_type is None:
				chunk_type, chunk_start = tok_chunk_type, i
			elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
				chunk = (chunk_type, chunk_start, i)
				chunks.append(chunk)
				chunk_type, chunk_start = tok_chunk_type, i
		else:
			pass
	# end condition
	if chunk_type is not None:
		chunk = (chunk_type, chunk_start, len(seq))
		chunks.append(chunk)

	return chunks








#   getAspectValue(test_word_sequences, test_trueTag_sequences, test_word_sequences_sent, dict_precomputed_path)

def getAspectValue(test_word_sequences, test_trueTag_sequences, test_word_sequences_sent,
				   test_trueTag_sequences_sent, dict_preComputed_path, dict_aspect_func):


	def getSententialValue(test_trueTag_sequences_sent, test_word_sequences_sent):

		eDen = []
		sentLen = []

		for i, test_sent in enumerate(test_trueTag_sequences_sent):
			pred_chunks = set(get_chunks(test_sent))

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
		if os.path.exists(path):
			print('load the hard dictionary of entity span in test set...')
			fread = open(path, 'rb')
			dict_preComputed_model[aspect] = pickle.load(fread)
		else:
			raise ValueError("can not load hard dictionary" + aspect + "\t" + path)




	dict_span2aspectVal = {}
	dict_chunkid2span = {}
	for aspect, fun in dict_aspect_func.items():
		dict_span2aspectVal[aspect] = {}

	eDen_list, sentLen_list = [], []
	eDen_list, sentLen_list = getSententialValue(test_trueTag_sequences_sent,
																	 test_word_sequences_sent)


	dict_pos2sid = getPos2SentId(test_word_sequences_sent)
	dict_ap2rp = getTokenPosition(test_word_sequences_sent)
	all_chunks = get_chunks(test_trueTag_sequences)
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
		dict_chunkid2span[span_pos] = format4json(span_cnt) + "|||" + format4json(' '.join(test_word_sequences_sent[span_sentid]))


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
	# path_true_file = args.true_file
	# path_pred_file = args.pred_file


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
	# print(list_text_token)
	list_true_tags_sent, list_true_tags_token = read_single_column(path_text, 1)
	list_pred_tags_sent, list_pred_tags_token = read_single_column(path_text, 2)

	#print(len(sum(list_text_sent,[])), len(list_text_token))


	#pred_chunks = set(get_chunks(list_true_tags_token[0:20]))
	#print(pred_chunks)

	dict_span2aspectVal, dict_span2sid, dict_chunkid2span  = getAspectValue(list_text_token, list_true_tags_token, list_text_sent, list_true_tags_sent, dict_preComputed_path, dict_aspect_func)
	dict_span2aspectVal_pred, dict_span2sid_pred, dict_chunkid2span_pred = getAspectValue(list_text_token, list_pred_tags_token, list_text_sent, list_pred_tags_sent, dict_preComputed_path, dict_aspect_func)


	holistic_performance = f1(list_true_tags_sent, list_pred_tags_sent)["f1"]
	confidence_low_overall, confidence_up_overall = compute_confidence_interval_f1(dict_span2sid.keys(), dict_span2sid_pred.keys(), dict_span2sid, dict_span2sid_pred, n_times=1000)

	# print(dict_span2aspectVal)

	print("confidence_low_overall:\t", confidence_low_overall)
	print("confidence_up_overall:\t", confidence_up_overall)
	# holistic_performance = f1(list_true_tags_sent, list_pred_tags_sent)["f1"]
	#print(f1(list_true_tags_sent, list_pred_tags_sent))





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
	errorCase_list = []

	for aspect, func in dict_aspect_func.items():
		# print(aspect, dict_span2aspectVal[aspect])
		dict_bucket2span[aspect] = __selectBucktingFunc(func[0], func[1], dict_span2aspectVal[aspect])
		# print(aspect, dict_bucket2span[aspect])
		# exit()
		dict_bucket2span_pred[aspect] = bucketAttribute_SpecifiedBucketInterval(dict_span2aspectVal_pred[aspect],
																				dict_bucket2span[aspect].keys())
		dict_bucket2f1[aspect], errorCase_list = getBucketF1_chunk(dict_bucket2span[aspect], dict_bucket2span_pred[aspect], dict_span2sid, dict_span2sid_pred, dict_chunkid2span, dict_chunkid2span_pred)
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
			bucket_value   = format(float(v[0])*100, '.4g')
			n_sample = v[1]
			confidence_low = format(float(v[2])*100, '.4g')
			confidence_up  = format(float(v[3])*100, '.4g')
			error_entity_list = v[4]
			# instantiation
			dict_fineGrained[aspect].append({"bucket_name":bucket_name, "bucket_value":bucket_value, "num":n_sample, "confidence_low":confidence_low, "confidence_up":confidence_up, "bucket_error_case":error_entity_list})





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

