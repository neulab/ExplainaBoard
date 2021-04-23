import numpy as np
import pickle
import codecs
import os
from collections import Counter
import re
import math
import scipy.stats as statss
import json
import random
import numpy
from ner_overall_f1 import evaluate,evaluate_chunk_level,evaluate_each_class,evaluate_each_class_listone
from seqeval.metrics import precision_score, recall_score, f1_score
#from sklearn.metrics import f1_score
from nltk.tokenize import TweetTokenizer
from collections import OrderedDict

from metrics.fragment import Fragments
from metrics.novelty_repetition_unk import *
from random import choices
import scipy.stats
import csv









def format4json(sent):
    sent = sent.replace(":"," ").replace("\"","").replace("\'","").replace("/","").replace("\\","").replace("{","").replace("}","")
    sent = sent.replace("\"","")
    return sent


def cap_feature(s):
    """
    Capitalization feature:
    0 = low caps
    1 = all caps
    2 = first letter caps
    3 = one capital (not first letter)
    """
    if s.lower() == s:
        return "low_caps"
    elif s.upper() == s:
        return "full_caps"
    elif s[0].upper() == s[0]:
        return "first_caps"
    else:
        return "not_first_caps"


def dict_char2word(sentence):
	ind_w = 0
	dict_c2w = {}
	for ind, c in enumerate(sentence):
		dict_c2w[ind] = ind_w
		if c ==" ":
			ind_w += 1
	return dict_c2w


def get_sample_rate(n_data):
	res = 0.8
	if n_data > 300000:
		res = 0.1
	elif n_data > 100000 and n_data < 300000:
		res = 0.2

	return res


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m-h, m+h


def compute_confidence_interval_acc(true_label_list, pred_label_list, n_times=1000):
	n_data = len(true_label_list)
	sample_rate = get_sample_rate(n_data)
	n_sampling = int(n_data * sample_rate)
	if n_sampling == 0:
	    n_sampling = 1
	print("n_data:\t", n_data)
	print("sample_rate:\t", sample_rate)
	print("n_sampling:\t", n_sampling)


	performance_list = []
	confidence_low, confidence_up = 0,0
	for i in range(n_times):
		sample_index_list = choices(range(n_data), k=n_sampling)

		performance = accuracy(list(np.array(true_label_list)[sample_index_list]),
							   list(np.array(pred_label_list)[sample_index_list]))
		performance_list.append(performance)


	if n_times != 1000:
		confidence_low, confidence_up = mean_confidence_interval(performance_list)
	else:
		performance_list.sort()
		confidence_low = performance_list[24]
		confidence_up = performance_list[974]


	print("\n")
	print("confidence_low:\t", confidence_low)
	print("confidence_up:\t", confidence_up)

	return confidence_low, confidence_up


# 1000
def compute_confidence_interval_f1(spans_true, spans_pred, dict_span2sid, dict_span2sid_pred, n_times=1000):
	n_data = len(dict_span2sid)
	sample_rate = get_sample_rate(n_data)
	n_sampling = int(n_data * sample_rate)
	print("sample_rate:\t", sample_rate)
	print("n_sampling:\t", n_sampling)



	dict_sid2span_salient = {}
	for span in spans_true:
		#print(span)
		if len(span.split("_"))!=3:
			break
		sid = dict_span2sid[span]
		if sid in dict_sid2span_salient.keys():
			dict_sid2span_salient[sid].append(span)
		else:
			dict_sid2span_salient[sid] = [span]

	dict_sid2span_salient_pred = {}
	for span in spans_pred:
		sid = dict_span2sid_pred[span]
		if sid in dict_sid2span_salient_pred.keys():
			dict_sid2span_salient_pred[sid].append(span)
		else:
			dict_sid2span_salient_pred[sid] = [span]





	performance_list = []
	confidence_low, confidence_up = 0,0
	for i in range(n_times):
		sample_index_list = choices(range(n_data), k=n_sampling)

		true_label_bootstrap_list = []
		pred_label_bootstrap_list = []
		for ind, sid in enumerate(sample_index_list):


			if sid in dict_sid2span_salient.keys():
				true_label_list = dict_sid2span_salient[sid]
				true_label_list_revised = [true_label + "_" + str(ind) for true_label in true_label_list]
				true_label_bootstrap_list += true_label_list_revised

			if sid in dict_sid2span_salient_pred.keys():
				pred_label_list = dict_sid2span_salient_pred[sid]
				pred_label_list_revised = [pred_label + "_" + str(ind) for pred_label in pred_label_list]
				pred_label_bootstrap_list += pred_label_list_revised

		f1, p, r = evaluate_chunk_level(pred_label_bootstrap_list, true_label_bootstrap_list)
		performance_list.append(f1)


	if n_times != 1000:
		confidence_low, confidence_up = mean_confidence_interval(performance_list)
	else:
		performance_list.sort()
		confidence_low = performance_list[24]
		confidence_up = performance_list[974]


	# print("\n")
	# print("confidence_low:\t", confidence_low)
	# print("confidence_up:\t", confidence_up)

	return confidence_low, confidence_up



################       Calculate Bucket-wise F1 Score:
def getBucketF1(dict_bucket2span, dict_bucket2span_pred, dict_span2sid, dict_span2sid_pred):
	print('------------------ attribute')
	dict_bucket2f1 = {}
	for bucket_interval, spans_true in dict_bucket2span.items():
		spans_pred = []


		#print('bucket_interval: ',bucket_interval)
		if bucket_interval not in dict_bucket2span_pred.keys():
			#print(bucket_interval)
			raise ValueError("Predict Label Bucketing Errors")
		else:
			spans_pred = dict_bucket2span_pred[bucket_interval]


		# print("debug----------")
		# print(len(dict_span2sid))
		# print(len(dict_span2sid_pred))

		confidence_low, confidence_up = compute_confidence_interval_f1(spans_true, spans_pred, dict_span2sid, dict_span2sid_pred)

		confidence_low = format(confidence_low , '.3g')
		confidence_up = format(confidence_up, '.3g')


		f1, p, r = evaluate_chunk_level(spans_pred, spans_true)
		print("-----------print spans_pred -------------")
		print(spans_pred)



		print("confidence_low:\t", confidence_low)
		print("confidence_up:\t", confidence_up)
		print("F1:\t", f1)

		print("------------------------------------------")



		dict_bucket2f1[bucket_interval] = [f1, len(spans_true), confidence_low, confidence_up]

		# if bucket_interval[0] == 1.0:
		# 	print("debug-f1:",f1)
		# 	print(spans_pred[0:20])
		# 	print(spans_true[0:20])
	# print("dict_bucket2f1: ",dict_bucket2f1)
	return sortDict(dict_bucket2f1)


# dict_chunkid2spanSent:  2_3 -> New York|||This is New York city
# dict_pos2tag: 2_3 -> NER
def get_errorCase(dict_pos2tag, dict_pos2tag_pred, dict_chunkid2spanSent, dict_chunkid2spanSent_pred):

    errorCase_list = []
    for pos, tag in dict_pos2tag.items():

        true_label = tag
        pred_label = ""
        #print(dict_chunkid2spanSent.keys())
        if pos+"_"+tag not in dict_chunkid2spanSent.keys():
            continue
        span_sentence = dict_chunkid2spanSent[pos+"_"+tag]

        if pos in dict_pos2tag_pred.keys():
            pred_label = dict_pos2tag_pred[pos]
            if true_label == pred_label:
                continue
        else:
            pred_label = "O"
        error_case = span_sentence + "|||" + true_label + "|||" + pred_label
        errorCase_list.append(error_case)


    for pos, tag in dict_pos2tag_pred.items():

        true_label = ""
        pred_label = tag
        if pos+"_"+tag not in dict_chunkid2spanSent_pred.keys():
            continue
        span_sentence = dict_chunkid2spanSent_pred[pos+"_"+tag]

        if pos in dict_pos2tag.keys():
            true_label = dict_pos2tag[pos]
            if true_label == pred_label:
                continue
        else:
            true_label = "O"
        error_case = span_sentence + "|||" + true_label + "|||" + pred_label
        errorCase_list.append(error_case)

    #print(errorCase_list)
    return errorCase_list




################       Calculate Bucket-wise F1 Score:
def getBucketF1_ner(dict_bucket2span, dict_bucket2span_pred, dict_span2sid, dict_span2sid_pred, dict_chunkid2span, dict_chunkid2span_pred):
	print('------------------ attribute')
	dict_bucket2f1 = {}




    # predict:  2_3 -> NER
	dict_pos2tag_pred = {}
	for k_bucket_eval, spans_pred in dict_bucket2span_pred.items():
		for span_pred in spans_pred:
			pos_pred = "_".join(span_pred.split("_")[0:2])
			tag_pred = span_pred.split("_")[-1]
			dict_pos2tag_pred[pos_pred] = tag_pred
		#print(dict_pos2tag_pred)

    # true:  2_3 -> NER
	dict_pos2tag = {}
	for k_bucket_eval, spans in dict_bucket2span.items():
		for span in spans:
			pos = "_".join(span.split("_")[0:2])
			tag = span.split("_")[-1]
			dict_pos2tag[pos] = tag
    # print(dict_pos2tag_pred)

	errorCase_list = get_errorCase(dict_pos2tag, dict_pos2tag_pred, dict_chunkid2span, dict_chunkid2span_pred)



	for bucket_interval, spans_true in dict_bucket2span.items():
		spans_pred = []


		#print('bucket_interval: ',bucket_interval)
		if bucket_interval not in dict_bucket2span_pred.keys():
			#print(bucket_interval)
			raise ValueError("Predict Label Bucketing Errors")
		else:
			spans_pred = dict_bucket2span_pred[bucket_interval]




		confidence_low, confidence_up = compute_confidence_interval_f1(spans_true, spans_pred, dict_span2sid, dict_span2sid_pred)

		confidence_low = format(confidence_low , '.3g')
		confidence_up = format(confidence_up, '.3g')


		f1, p, r = evaluate_chunk_level(spans_pred, spans_true)





		print("-----------print spans_pred -------------")

		error_entity_list = []
		for span_true in spans_true:
		    if span_true not in spans_pred:
		        #print(span_true)
		        pos_true = "_".join(span_true.split("_")[0:2])
		        tag_true = span_true.split("_")[-1]

		        if pos_true in dict_pos2tag_pred.keys():
		            tag_pred = dict_pos2tag_pred[pos_true]
		            if tag_pred != tag_true:
		                error_entity_list.append(dict_chunkid2span[span_true] + "|||" + tag_true + "|||" + dict_pos2tag_pred[pos_true])
		        else:
		            error_entity_list.append(dict_chunkid2span[span_true] + "|||" + tag_true + "|||" + "O")



		print("confidence_low:\t", confidence_low)
		print("confidence_up:\t", confidence_up)
		print("F1:\t", f1)
		#print(error_entity_list)

		print("------------------------------------------")



		dict_bucket2f1[bucket_interval] = [f1, len(spans_true), confidence_low, confidence_up, error_entity_list]

		# if bucket_interval[0] == 1.0:
		# 	print("debug-f1:",f1)
		# 	print(spans_pred[0:20])
		# 	print(spans_true[0:20])
	# print("dict_bucket2f1: ",dict_bucket2f1)
	return sortDict(dict_bucket2f1), errorCase_list


def getBucketAcc(dict_bucket2span, dict_bucket2span_pred):
	print('------------------ attribute')
	dict_bucket2f1 = {}
	for bucket_interval, spans_true in dict_bucket2span.items():
		spans_pred = []

		print('bucket_interval: ',bucket_interval)
		if bucket_interval not in dict_bucket2span_pred.keys():
			#print(bucket_interval)
			raise ValueError("Predict Label Bucketing Errors")
		else:
			spans_pred = dict_bucket2span_pred[bucket_interval]


		accuracy_each_bucket = accuracy(spans_pred, spans_true)
		confidence_low, confidence_up = compute_confidence_interval_acc(spans_pred, spans_true)
		dict_bucket2f1[bucket_interval] = [accuracy_each_bucket, len(spans_true), confidence_low, confidence_up]




		print("accuracy_each_bucket:\t", accuracy_each_bucket)

	return sortDict(dict_bucket2f1)



################       Calculate Bucket-wise F1 Score:

def getBucketROUGE(dict_bucket2span):
	print('------------------ attribute')
	dict_bucket2f1 = {}
	for bucket_interval, spans_true in dict_bucket2span.items():
		spans_pred = []

		rouge_list = [float(sample_pos.split("_")[-1]) for sample_pos in spans_true]
		avg_rouge = np.average(rouge_list)

		print('bucket_interval: ',bucket_interval)


		dict_bucket2f1[bucket_interval] = [avg_rouge, len(spans_true)]

	return sortDict(dict_bucket2f1)



def compute_holistic_f1_re(path, delimiter = "\t"):
	fin = open(path, "r")
	true_list = []
	pred_list = []
	for line in fin:
		if len(line.split("\t"))!=3:
			#print(line)
			continue
		line = line.rstrip()
		true_list.append(line.split("\t")[-2])
		pred_list.append(line.split("\t")[-1])
	f1 = f1_score(true_list, pred_list, average='micro')
	# print(true_list[0:10])
	# print(pred_list[0:10])
	# print("------f1-----------")
	# print(f1)
	#exit()
	return f1


def compute_holistic_f1(fn_result, delimiter = " "):
	if delimiter == " ":
		cmd = 'perl  %s -d \"\t\" < %s' % (os.path.join('.', 'conlleval'), fn_result)

	msg = '\nStandard CoNNL perl script (author: Erik Tjong Kim Sang <erikt@uia.ua.ac.be>, version: 2004-01-26):\n'
	msg += ''.join(os.popen(cmd).readlines())
	print("result: ",msg)
	f1 = float(msg.split('\n')[3].split(':')[-1].strip())

	return f1



def accuracy(labels, predictions, language=None):
  correct = sum([int(p == l) for p, l in zip(predictions, labels)])
  accuracy = float(correct) / len(predictions)
  return accuracy*100




def get_ci_interval(confidence_val, confidence_delta):
	info = "(" + str(confidence_val) + "-" + str(confidence_delta) + ", " + str(confidence_val) + "+" + str(
		confidence_delta) + ")"
	return info

def distance(text_sents, summary_sents):
	density, coverage, compression, copy_len, novelty_1, novelty_2, repetition_1, repetition_2 = 0,0,0,0,0,0,0,0

	fragment = Fragments("\n".join(summary_sents), " ".join(text_sents))
	compression = len(text_sents.split(" "))/len(summary_sents.split(" "))
	density = fragment.density()
	# coverage = fragment.coverage()
	# compression = fragment.compression()
	copy_len = 0 if len(fragment.copy_len()) == 0 else sum(fragment.copy_len()) / len(fragment.copy_len())

	novelty_1 = novelty_oneSample(text_sents, summary_sents, 1)
	novelty_2 = novelty_oneSample(text_sents, summary_sents, 2)

	repetition_1 = repetition_oneSample(summary_sents, 1)
	# repetition_2 = repetition_oneSample(summary_sents, 2)


	print(density, coverage, compression, copy_len, novelty_1, novelty_2, repetition_1, repetition_2)

	return density, coverage, compression, copy_len, novelty_1, novelty_2, repetition_1, repetition_2


def list_minus(a, b):
    return [tmpa - tmpb for tmpa, tmpb in zip(a, b)]


def get_avg(res):
    result = {}
    for key, value in res.items():
        if isinstance(value, list):
            result[key] = sum(value) / len(value)
        else:
            result[key] = value
    return result


def wordSegment2(sent):
	tknzr = TweetTokenizer()
	token_list = tknzr.tokenize(sent)
	return token_list



def wordSegment(sent):
	if len(sent.split(" ")) == 1 and len(list(sent)) >= 10:
		return " ".join(list(sent))
	else:
		return sent

def intervalTransformer(inter_list):
	dict_old2new = {}
	last = 0
	for ind, interval in enumerate(inter_list):
		if ind == 0:
			last = interval[0]
		if len(interval) == 1:
			#new_inter_list.append(interval)
			dict_old2new[interval] = interval
			last = interval[0]
		else:
			#new_inter_list.append((last, interval[1]))
			dict_old2new[interval] = (last, interval[1])
			last = interval[1]
	return dict_old2new



def sortDict(dict_obj, flag = "key"):
	sorted_dict_obj  = []
	if flag == "key":
		sorted_dict_obj = sorted(dict_obj.items(), key=lambda item:item[0])
	elif flag == "value":
		#dict_bucket2span_
		sorted_dict_obj = sorted(dict_obj.items(), key=lambda item:len(item[1]), reverse = True)
	return dict(sorted_dict_obj)



def reverseDict(dict_a2b):
	dict_b2a = {}
	for k, v in dict_a2b.items():
		v = float(v)
		if v not in dict_b2a.keys():
			dict_b2a[float(v)] = [k]
		else:
			dict_b2a[float(v)].append(k)


	return dict_b2a

def reverseDict_discrete(dict_a2b):
	dict_b2a = {}
	for k, v in dict_a2b.items():
		if v not in dict_b2a.keys():
			dict_b2a[v] = [k]
		else:
			dict_b2a[v].append(k)


	return dict_b2a


def findKey(dict_obj, x):
	for k, v in dict_obj.items():
			if len(k) == 1:
				if x == k[0]:
					return k
			elif len(k) ==2 and x >= k[0] and x <= k[1]:  # Attention !!!
					return k



def tuple2str(triplet):
	res = ""
	for v in triplet:
		res += str(v) + "|||"
	return res.rstrip("|||")





def bucketAttribute_SpecifiedBucketValue(dict_span2attVal, n_buckets, hardcoded_bucket_values):
		################       Bucketing different Attributes


		# hardcoded_bucket_values = [set([float(0), float(1)])]
		#print("!!!debug-7--")
		p_infinity = 1000000
		n_infinity = -1000000
		n_spans = len(dict_span2attVal)
		dict_attVal2span = reverseDict(dict_span2attVal)
		dict_attVal2span = sortDict(dict_attVal2span)
		dict_bucket2span = {}



		for backet_value in hardcoded_bucket_values:
			if backet_value in dict_attVal2span.keys():
				#print("------------work!!!!---------")
				#print(backet_value)
				dict_bucket2span[(backet_value,)] = dict_attVal2span[backet_value]
				n_spans -= len(dict_attVal2span[backet_value])
				n_buckets -= 1


		avg_entity = n_spans * 1.0 / n_buckets
		n_tmp = 0
		entity_list = []
		val_list = []

		#
		#print("-----avg_entity----------")
		#print(avg_entity)


		for attval, entity in dict_attVal2span.items():
			if attval in hardcoded_bucket_values:
				continue


			# print("debug-attval:\t",attval)
			val_list.append(attval)
			entity_list += entity
			n_tmp += len(entity)

			# print(attval)
			# print(n_tmp, avg_entity)

			if n_tmp > avg_entity:

				if len(val_list) >=2:
					key_bucket = (val_list[0], val_list[-1])
					dict_bucket2span[key_bucket] = entity_list
					#print("debug key bucket:\t", key_bucket)
				else:
					dict_bucket2span[(val_list[0],)] = entity_list
				entity_list = []
				n_tmp = 0
				val_list = []
		if n_tmp != 0:
			if n_buckets == 1:
				dict_bucket2span[(n_infinity,p_infinity)] = entity_list
			else:
				if val_list[0] <=1:
					p_infinity = 1.0
				#print("!!!!!-debug-2")
				if len(val_list) >=2:
					key_bucket = (val_list[0], p_infinity)
					dict_bucket2span[key_bucket] = entity_list
				else:
					dict_bucket2span[(val_list[0],p_infinity)] = entity_list # fix bugs
		#
		#
		#
		# [(0,), (0.1, 0.2), (0.3,0.4), (0.5, 0.6)] --> [(0,), (0,0.2), (0.2, 0.4), (0.4, 0.6)]
		# dict_old2new = intervalTransformer(dict_bucket2span.keys())
		# dict_bucket2span_new = {}
		# for inter_list, span_list in dict_bucket2span.items():
		# 	dict_bucket2span_new[dict_old2new[inter_list]] = span_list

		return dict_bucket2span







def bucketAttribute_DiscreteValue(dict_span2attVal = None, n_buckets = 100000000, n_entities = 1):
	################          Bucketing different Attributes


	#print("!!!!!debug---------")
	# 	hardcoded_bucket_values = [set([float(0), float(1)])]
	n_spans = len(dict_span2attVal)
	dict_bucket2span = {}

	dict_attVal2span = reverseDict_discrete(dict_span2attVal)
	dict_attVal2span = sortDict(dict_attVal2span, flag = "value")

	# dict["q_id"] = 2



	avg_entity = n_spans * 1.0 / n_buckets
	n_tmp = 0
	entity_list = []
	val_list = []



	n_total = 1
	for attval, entity in dict_attVal2span.items():

		if len(entity) < n_entities or n_total > n_buckets:
			break
		dict_bucket2span[(attval,)] = entity

		n_total += 1

	return dict_bucket2span






def bucketAttribute_SpecifiedBucketInterval(dict_span2attVal, intervals):
	################       Bucketing different Attributes

	#hardcoded_bucket_values = [set([float(0), float(1)])]

	#intervals = [0, (0,0.5], (0.5,0.9], (0.99,1]]

	dict_bucket2span = {}
	n_spans = len(dict_span2attVal)

	#print("!!!!!!!enter into bucketAttribute_SpecifiedBucketInterval")

	#print(intervals)

	if type(list(intervals)[0][0]) == type("string"):  # discrete value, such as entity tags
		dict_attVal2span = reverseDict_discrete(dict_span2attVal)
		dict_attVal2span = sortDict(dict_attVal2span, flag = "value")
		for attval, entity in dict_attVal2span.items():
			attval_tuple = (attval,)
			if attval_tuple in intervals:
				if attval_tuple not in dict_bucket2span.keys():
					dict_bucket2span[attval_tuple] = entity
				else:
					dict_bucket2span[attval_tuple] += entity

		for val in intervals:
			if val not in dict_bucket2span.keys():
				dict_bucket2span[val] = []
		# print("dict_bucket2span: ",dict_bucket2span)
	else:
		#print("---debug----5")
		#print(intervals)
		dict_attVal2span = reverseDict(dict_span2attVal)
		dict_attVal2span = sortDict(dict_attVal2span)
		for v in intervals:
			if len(v) == 1:
				dict_bucket2span[v] = []
			else:
				dict_bucket2span[v] = []

		# print("debug-interval:\t", intervals)

		for attval, entity in dict_attVal2span.items():
					res_key = findKey(dict_bucket2span, attval)
					#print("res-key:\t"+ str(res_key))
					if res_key == None:
						continue
					dict_bucket2span[res_key] += entity

	return dict_bucket2span











def printDict(dict_obj, info="dict"):
	#print("-----------------------------------------------")
	print("the information of #" + info + "#")
	print("Bucket_interval\tF1\tEntity-Number")
	for k,v in dict_obj.items():
		if len(k) == 1:
			print("[" + str(k[0])+",]" + "\t" + str(v[0]) + "\t" + str(v[1]))
		else:
			print("[" + str(k[0])+", " + str(k[1]) +"]" + "\t" + str(v[0]) + "\t" + str(v[1]))

	print("")
def extValue(cont, fr, to):
	return cont.split(fr)[-1].split(to)[0]


def loadConf(path_conf):
	fin = open(path_conf,"r")
	all_cont = fin.read()
	dict_aspect_func={}
	for block in all_cont.split("# "):
		# print("debug3-------")
		# print(block)
		notation = extValue(block, "notation:\t", "\n").rstrip(" ")
		if notation == "":
			continue
		# print("debug4--notation-----")
		# print(notation)
		func_type = extValue(block, "type:\t", "\n").rstrip(" ")
		func_setting = extValue(block, "setting:\t", "\n").rstrip(" ")
		is_preComputed = extValue(block, "is_preComputed:\t", "\n").rstrip(" ")
		dict_aspect_func[notation] = (func_type, func_setting, is_preComputed)
	# exit()
	return dict_aspect_func




def ensureDir(f):
	if not os.path.exists(f):
		os.makedirs(f)



def load_json(path):
	with open(path, "r") as f:
		json_template = json.load(f)
	# steps = [Step.from_dict(step_dict) for step_dict in schemas["steps"]]
	return json_template


def save_json(obj_json, path):
	with open(path, "w") as f:
		json.dump(obj_json, f, indent=4, ensure_ascii=False)


def getPos2SentId(test_word_sequences_sent):
	dict_pos2sid = {}
	pos = 0
	for sid, sent in enumerate(test_word_sequences_sent):
		for i in range(len(sent)):
			dict_pos2sid[pos] = sid
			pos += 1
	return dict_pos2sid


def getTokenPosition(test_word_sequences_sent):
	dict_ap2rp = {}
	pos = 0
	for sid, sent in enumerate(test_word_sequences_sent):
		for i in range(len(sent)):
			dict_ap2rp[pos] = i
			pos += 1
	return dict_ap2rp


def file2list(path_file):
	res_list = []
	fin = open(path_file,"r")
	for line in fin:
		line = line.rstrip("\n")
		res_list.append(line)
	fin.close()
	return res_list


def file_to_list_triple(path_file):
	sent_list = []
	true_label_list = []
	pred_label_list = []
	fin = open(path_file,"r")
	for line in fin:
		line = line.rstrip("\n")
		if len(line.split("\t")) !=3:
			continue
		sent, true_label, pred_label = line.split("\t")[0], line.split("\t")[1], line.split("\t")[2]
		sent_list.append(sent)
		true_label_list.append(true_label)
		pred_label_list.append(pred_label)

	fin.close()
	return sent_list, true_label_list, pred_label_list

def file_to_list_tc(path_file):
	sent_list = []
	true_label_list = []
	pred_label_list = []
	fin = open(path_file,"r")
	for line in fin:
		line = line.rstrip("\n")
		if len(line.split("\t")) !=5:
			continue
		sent, true_label, pred_label = line.split("\t")[0], line.split("\t")[1], line.split("\t")[2]
		sent_list.append(sent)
		true_label_list.append(true_label)
		pred_label_list.append(pred_label)

	fin.close()
	return sent_list, true_label_list, pred_label_list



def file_to_list_nli(path_file):
	sent1_list = []
	sent2_list = []
	true_label_list = []
	pred_label_list = []
	fin = open(path_file,"r")
	for line in fin:
		line = line.rstrip("\n")
		if len(line.split("\t")) <4:
			continue
		sent1, sent2, true_label, pred_label = line.split("\t")[0], line.split("\t")[1], line.split("\t")[2], line.split("\t")[3]
		sent1_list.append(sent1)
		sent2_list.append(sent2)
		true_label_list.append(true_label)
		pred_label_list.append(pred_label)

	fin.close()
	return sent1_list, sent2_list, true_label_list, pred_label_list



def file_to_list_absa(path_file):
	sent1_list = []
	sent2_list = []
	true_label_list = []
	pred_label_list = []
	fin = open(path_file,"r")
	for line in fin:
		line = line.rstrip("\n")
		if len(line.split("\t")) <4:
			continue
		sent1, sent2, true_label, pred_label = line.split("\t")[0], line.split("\t")[1], line.split("\t")[2], line.split("\t")[3]
		sent1_list.append(sent1)
		sent2_list.append(sent2)
		true_label_list.append(true_label)
		pred_label_list.append(pred_label)

	fin.close()
	return sent1_list, sent2_list, true_label_list, pred_label_list



def file_to_list_summ(path_file):
	doc_list = []
	hyp_list = []
	ref_list = []
	r1 = []
	r2 = []
	rl = []
	r1_overall = []
	r2_overall = []
	rl_overall = []
	fin = open(path_file,"r")
	for line in fin:
		line = line.rstrip("\n")
		if len(line.split("\t")) <9:
			continue
		sent, true_label, pred_label = line.split("\t")[0], line.split("\t")[1], line.split("\t")[2]
		doc_list.append(line.split("\t")[0])
		hyp_list.append(line.split("\t")[1])
		ref_list.append(line.split("\t")[2])
		r1.append(line.split("\t")[3])
		r2.append(line.split("\t")[4])
		rl.append(line.split("\t")[5])
		r1_overall.append(line.split("\t")[6])
		r2_overall.append(line.split("\t")[7])
		rl_overall.append(line.split("\t")[8])

	fin.close()
	return doc_list, hyp_list, ref_list, r1, r2, rl, r1_overall, r2_overall, rl_overall








def file2listPair(path_file):
	sent1_list = []
	sent2_list = []
	fin = open(path_file,"r")
	for line in fin:
		line = line.rstrip("\n")
		sent1, sent2 = line.split("\t")[0], line.split("\t")[1]
		sent1_list.append(sent1)
		sent2_list.append(sent2)

	fin.close()
	return sent1_list, sent2_list



def file2list_firstColumn(path_file):
	res_list = []
	fin = open(path_file,"r")
	for line in fin:
		line = line.rstrip("\n").split("\t")[0]
		res_list.append(line)
	fin.close()
	return res_list



def file2dict(path_file):
	res_dict = {}
	fin = open(path_file,"r")
	for line in fin:
		line = line.rstrip("\n")
		sent_id, sent = line.split("\t")
		res_dict[sent_id] = sent

	fin.close()
	return res_dict


def read_tag_pos(file):
  labels = []
  example = []
  labels_holistic = []
  with open(file, 'r') as f:
    for line in f:
      line = line.strip()
      if line:
        example.append("B-"+line)
        #print("B"+line)
        labels_holistic.append("B-"+line)
      else:
        labels.append(example)
        example = []
  if example:
    labels.append(example)
  return labels, labels_holistic


# def read_tag(file):
#   labels = []
#   example = []
#   with open(file, 'r') as f:
#     for line in f:
#       line = line.strip()
#       if line:
#         example.append(line)
#       else:
#         labels.append(example)
#         example = []
#   if example:
#     labels.append(example)
#   return labels



def read_text_pos(file):
  labels = []
  example = []
  labels_holistic = []
  with open(file, 'r') as f:
    for line in f:
      line = line.strip()
      if line:
        line = line.split("\t")[0]
        example.append(line)
        labels_holistic.append(line)
      else:
        labels.append(example)
        example = []
  if example:
    labels.append(example)
  return labels, labels_holistic




def read_tag(file):
  labels = []
  example = []
  labels_holistic = []
  with open(file, 'r') as f:
    for line in f:
      line = line.strip()
      if line:
        example.append(line)
        labels_holistic.append(line)
      else:
        labels.append(example)
        example = []
  if example:
    labels.append(example)
  return labels, labels_holistic



def read_single_column(file,k):
  labels = []
  example = []
  labels_holistic = []
  with open(file, 'r') as f:
    for line in f:
      line = line.strip()
      if line:
        if len(line.split("\t")) !=3:
           print(line)
        example.append(line.split("\t")[k])
        labels_holistic.append(line.split("\t")[k])
      else:
        labels.append(example)
        example = []
  if example:
    labels.append(example)
  return labels, labels_holistic











def bucc_f1(labels, predictions, language=None):
  labels = set([tuple(l.split('\t')) for l in labels])
  predictions = set([tuple(l.split('\t')) for l in predictions])
  ncorrect = len(labels.intersection(predictions))
  if ncorrect > 0:
    precision = ncorrect / len(predictions)
    recall = ncorrect / len(labels)
    f1 = 2 * precision * recall / (precision + recall)
  else:
    precision = recall = f1 = 0
  return {'f1': f1 * 100, 'precision': precision * 100, 'recall': recall * 100}


def f1(labels, predictions, language=None):
  f1 = f1_score(labels, predictions)
  precision = precision_score(labels, predictions)
  recall = recall_score(labels, predictions)
  return {'f1': f1 * 100, 'precision': precision * 100, 'recall': recall * 100}




