from utils import *



def format4json_tc(sent):
    sent = sent.replace(":"," ").replace("\"","").replace("\'","").replace("/","").replace("\\","").replace("{","").replace("}","")
    sent = sent.replace("\"","").replace("\\n","").replace("\\n\\n","").replace("\\\"\"\"","")

    if len(sent.split(" ")) > 521:
        wordlist = sent.split(" ")[:520]
        sent = " ".join(wordlist) + " ... "

    return sent

def getErrorCase_tc(sent_list, true_label_list, pred_label_list):
    errorCase_list = []
    for sent, true_label, pred_label in zip(sent_list, true_label_list, pred_label_list):
        if true_label != pred_label:
            errorCase_list.append(true_label + "|||" + pred_label +"|||" + format4json_tc(sent))
    return errorCase_list






def getBucketAcc_with_errorCase(dict_bucket2span, dict_bucket2span_pred, dict_sid2sent):


    # The structure of span_true or span_pred
    # 2345|||Positive
    # 2345 represents sentence id
    # Positive represents the "label" of this instance

    dict_bucket2f1 = {}


    for bucket_interval, spans_true in dict_bucket2span.items():
        spans_pred = []


        # print('bucket_interval: ',bucket_interval)
        if bucket_interval not in dict_bucket2span_pred.keys():
            #print(bucket_interval)
            raise ValueError("Predict Label Bucketing Errors")
        else:
            spans_pred = dict_bucket2span_pred[bucket_interval]



        # loop over samples from a given bucket
        error_case_bucket_list = []
        for info_true, info_pred in zip(spans_true, spans_pred):
            sid_true, label_true = info_true.split("|||")
            sid_pred, label_pred = info_pred.split("|||")
            if sid_true != sid_pred:
                continue

            sent = dict_sid2sent[sid_true]
            if label_true != label_pred:
                error_case_info  =  label_true + "|||" + label_pred + "|||" + sent
                error_case_bucket_list.append(error_case_info)


        accuracy_each_bucket = accuracy(spans_pred, spans_true)
        # print("debug: span_pred:\t")
        # print(spans_pred)
        confidence_low, confidence_up = compute_confidence_interval_acc(spans_pred, spans_true)
        dict_bucket2f1[bucket_interval] = [accuracy_each_bucket, len(spans_true), confidence_low, confidence_up, error_case_bucket_list]

        # print(error_case_bucket_list)

        print("accuracy_each_bucket:\t", accuracy_each_bucket)

    return sortDict(dict_bucket2f1)



def getErrorCase_nli(sent1_list, sent2_list, true_label_list, pred_label_list):
    errorCase_list = []
    for sent1, sent2, true_label, pred_label in zip(sent1_list, sent2_list, true_label_list, pred_label_list):
        if true_label != pred_label:
            errorCase_list.append(true_label + "|||" + pred_label +"|||" + format4json_tc(sent1) +"|||" + format4json_tc(sent2))
    return errorCase_list





def getBucketAcc_with_errorCase_nli(dict_bucket2span, dict_bucket2span_pred, dict_sid2sentpair):


    # The structure of span_true or span_pred
    # 2345|||Positive
    # 2345 represents sentence id
    # Positive represents the "label" of this instance

    dict_bucket2f1 = {}


    for bucket_interval, spans_true in dict_bucket2span.items():
        spans_pred = []


        # print('bucket_interval: ',bucket_interval)
        if bucket_interval not in dict_bucket2span_pred.keys():
            #print(bucket_interval)
            raise ValueError("Predict Label Bucketing Errors")
        else:
            spans_pred = dict_bucket2span_pred[bucket_interval]



        # loop over samples from a given bucket
        error_case_bucket_list = []
        for info_true, info_pred in zip(spans_true, spans_pred):
            sid_true, label_true = info_true.split("|||")
            sid_pred, label_pred = info_pred.split("|||")
            if sid_true != sid_pred:
                continue

            sent = dict_sid2sentpair[sid_true]
            if label_true != label_pred:
                error_case_info  =  label_true + "|||" + label_pred + "|||" + sent
                error_case_bucket_list.append(error_case_info)


        accuracy_each_bucket = accuracy(spans_pred, spans_true)
        # print("debug: span_pred:\t")
        # print(spans_pred)
        confidence_low, confidence_up = compute_confidence_interval_acc(spans_pred, spans_true)
        dict_bucket2f1[bucket_interval] = [accuracy_each_bucket, len(spans_true), confidence_low, confidence_up, error_case_bucket_list]

        # print(error_case_bucket_list)

        print("accuracy_each_bucket:\t", accuracy_each_bucket)

    return sortDict(dict_bucket2f1)




def getErrorCase_absa(aspect_list, sent_list, true_label_list, pred_label_list):
    errorCase_list = []
    for aspect, sent, true_label, pred_label in zip(aspect_list, sent_list, true_label_list, pred_label_list):
        if true_label != pred_label:
            errorCase_list.append(true_label + "|||" + pred_label +"|||" + format4json_tc(aspect) +"|||" + format4json_tc(sent))
    return errorCase_list


def getBucketAcc_with_errorCase_absa(dict_bucket2span, dict_bucket2span_pred, dict_sid2sentpair):


    # The structure of span_true or span_pred
    # 2345|||Positive
    # 2345 represents sentence id
    # Positive represents the "label" of this instance

    dict_bucket2f1 = {}


    for bucket_interval, spans_true in dict_bucket2span.items():
        spans_pred = []


        # print('bucket_interval: ',bucket_interval)
        if bucket_interval not in dict_bucket2span_pred.keys():
            #print(bucket_interval)
            raise ValueError("Predict Label Bucketing Errors")
        else:
            spans_pred = dict_bucket2span_pred[bucket_interval]



        # loop over samples from a given bucket
        error_case_bucket_list = []
        for info_true, info_pred in zip(spans_true, spans_pred):
            sid_true, label_true = info_true.split("|||")
            sid_pred, label_pred = info_pred.split("|||")
            if sid_true != sid_pred:
                continue

            sent = dict_sid2sentpair[sid_true]
            if label_true != label_pred:
                error_case_info  =  label_true + "|||" + label_pred + "|||" + sent
                error_case_bucket_list.append(error_case_info)


        accuracy_each_bucket = accuracy(spans_pred, spans_true)
        # print("debug: span_pred:\t")
        # print(spans_pred)
        confidence_low, confidence_up = compute_confidence_interval_acc(spans_pred, spans_true)
        dict_bucket2f1[bucket_interval] = [accuracy_each_bucket, len(spans_true), confidence_low, confidence_up, error_case_bucket_list]

        # print(error_case_bucket_list)

        print("accuracy_each_bucket:\t", accuracy_each_bucket)

    return sortDict(dict_bucket2f1)




# 1000
def compute_confidence_interval_f1_cws(spans_true, spans_pred, dict_span2sid, dict_span2sid_pred, n_times=1000):
	n_data = len(dict_span2sid)
	sample_rate = get_sample_rate(n_data)
	n_sampling = int(n_data * sample_rate)
	print("sample_rate:\t", sample_rate)
	print("n_sampling:\t", n_sampling)



	dict_sid2span_salient = {}
	for span in spans_true:
		#print(span)
		if len(span.split("|||"))!=3:
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
				true_label_list_revised = [true_label + "|||" + str(ind) for true_label in true_label_list]
				true_label_bootstrap_list += true_label_list_revised

			if sid in dict_sid2span_salient_pred.keys():
				pred_label_list = dict_sid2span_salient_pred[sid]
				pred_label_list_revised = [pred_label + "|||" + str(ind) for pred_label in pred_label_list]
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





# dict_chunkid2spanSent:  2_3 -> New York|||This is New York city
# dict_pos2tag: 2_3 -> NER
def get_errorCase_cws(dict_pos2tag, dict_pos2tag_pred, dict_chunkid2spanSent, dict_chunkid2spanSent_pred, list_true_tags_token, list_pred_tags_token):


	errorCase_list = []
	for pos, tag in dict_pos2tag.items():

		true_label = tag
		pred_label = ""
		#print(dict_chunkid2spanSent.keys())
		if pos+"|||"+tag not in dict_chunkid2spanSent.keys():
			continue
		span_sentence = dict_chunkid2spanSent[pos+"|||"+tag]

		if pos in dict_pos2tag_pred.keys():
			pred_label = dict_pos2tag_pred[pos]
			if true_label == pred_label:
				continue
			# print(pos + "\t" + true_label + "\t" + pred_label)
		else:
			start = int(pos.split("|||")[0])
			end = int(pos.split("|||")[1])
			pred_label = "".join(list_pred_tags_token[start:end])
			# print(pred_label)
		error_case = span_sentence + "|||" + true_label + "|||" + pred_label
		errorCase_list.append(error_case)


	for pos, tag in dict_pos2tag_pred.items():

		true_label = ""
		pred_label = tag
		if pos+"|||"+tag not in dict_chunkid2spanSent_pred.keys():
			continue
		span_sentence = dict_chunkid2spanSent_pred[pos+"|||"+tag]

		if pos in dict_pos2tag.keys():
			true_label = dict_pos2tag[pos]
			if true_label == pred_label:
				continue
		else:
			start = int(pos.split("|||")[0])
			end = int(pos.split("|||")[1])
			true_label = "".join(list_true_tags_token[start:end])
		error_case = span_sentence + "|||" + true_label + "|||" + pred_label
		errorCase_list.append(error_case)


	# for v in errorCase_list:
	# 	print(len(errorCase_list))
	# 	print(v)
	#print(errorCase_list)

	return errorCase_list


################       Calculate Bucket-wise F1 Score:
def getBucketF1_cws(dict_bucket2span, dict_bucket2span_pred, dict_span2sid, dict_span2sid_pred, dict_chunkid2span, dict_chunkid2span_pred, list_true_tags_token, list_pred_tags_token):
	print('------------------ attribute')
	dict_bucket2f1 = {}




    # predict:  2_3 -> NER
	dict_pos2tag_pred = {}
	for k_bucket_eval, spans_pred in dict_bucket2span_pred.items():
		for span_pred in spans_pred:
			pos_pred = "|||".join(span_pred.split("|||")[0:2])
			tag_pred = span_pred.split("|||")[-1]
			dict_pos2tag_pred[pos_pred] = tag_pred

			# for k, v in dict_pos2tag_pred.items():
			# 	if int(k.split("|||")[1]) - int(k.split("|||")[0]) != len(v):
			# 		print(k + "\t" + v)
		#print(dict_pos2tag_pred)

    # true:  2_3 -> NER
	dict_pos2tag = {}
	for k_bucket_eval, spans in dict_bucket2span.items():
		for span in spans:
			pos = "|||".join(span.split("|||")[0:2])
			tag = span.split("|||")[-1]
			dict_pos2tag[pos] = tag

		# for k, v in dict_pos2tag_pred.items():
		# 	if int(k.split("_")[1]) - int(k.split("_")[0])  != len(v):
		# 		print(k + "\t" + v)





	errorCase_list = get_errorCase_cws(dict_pos2tag, dict_pos2tag_pred, dict_chunkid2span, dict_chunkid2span_pred, list_true_tags_token, list_pred_tags_token)

	# print(len(errorCase_list))
	# print(errorCase_list)

	for bucket_interval, spans_true in dict_bucket2span.items():
		spans_pred = []


		#print('bucket_interval: ',bucket_interval)
		if bucket_interval not in dict_bucket2span_pred.keys():
			#print(bucket_interval)
			raise ValueError("Predict Label Bucketing Errors")
		else:
			spans_pred = dict_bucket2span_pred[bucket_interval]




		confidence_low, confidence_up = compute_confidence_interval_f1_cws(spans_true, spans_pred, dict_span2sid, dict_span2sid_pred)

		confidence_low = format(confidence_low , '.3g')
		confidence_up = format(confidence_up, '.3g')


		f1, p, r = evaluate_chunk_level(spans_pred, spans_true)





		print("-----------print spans_pred -------------")

		error_entity_list = []
		for span_true in spans_true:
			if span_true not in spans_pred:
				#print(span_true)
				pos_true = "|||".join(span_true.split("|||")[0:2])
				tag_true = span_true.split("|||")[-1]

				if pos_true in dict_pos2tag_pred.keys():
					tag_pred = dict_pos2tag_pred[pos_true]
					if tag_pred != tag_true:
						error_entity_list.append(dict_chunkid2span[span_true] + "|||" + tag_true + "|||" + dict_pos2tag_pred[pos_true])
						#print(dict_chunkid2span[span_true] + "|||" + tag_true + "|||" + dict_pos2tag_pred[pos_true])
				else:
					start = int(pos_true.split("|||")[0])
					end = int(pos_true.split("|||")[1])
					pred_label = "".join(list_pred_tags_token[start:end])
					error_entity_list.append(dict_chunkid2span[span_true] + "|||" + tag_true + "|||" + pred_label)

					#print(dict_chunkid2span[span_true] + "|||" + tag_true + "|||" + pred_label)



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



# dict_chunkid2spanSent:  2_3 -> New York|||This is New York city
# dict_pos2tag: 2_3 -> NER
def get_errorCase_pos(dict_pos2tag, dict_pos2tag_pred, dict_chunkid2spanSent, dict_chunkid2spanSent_pred):

	# print("debug-1:")
	# print()

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
			#pred_label = "O"
			continue


		error_case = format4json_tc(span_sentence) + "|||" + true_label + "|||" + pred_label

		# if pred_label == "O":
		# 	print(error_case)
		# 	print(len(dict_pos2tag), len(dict_pos2tag_pred))
		# 	print(pos)

		errorCase_list.append(error_case)


	#print(errorCase_list)
	return errorCase_list



# 1000
def compute_confidence_interval_f1_pos(spans_true, spans_pred, dict_span2sid, dict_span2sid_pred, n_times=100):
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




def getBucketF1_pos(dict_bucket2span, dict_bucket2span_pred, dict_span2sid, dict_span2sid_pred, dict_chunkid2span, dict_chunkid2span_pred):
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

	errorCase_list = get_errorCase_pos(dict_pos2tag, dict_pos2tag_pred, dict_chunkid2span, dict_chunkid2span_pred)



	for bucket_interval, spans_true in dict_bucket2span.items():
		spans_pred = []


		#print('bucket_interval: ',bucket_interval)
		if bucket_interval not in dict_bucket2span_pred.keys():
			#print(bucket_interval)
			raise ValueError("Predict Label Bucketing Errors")
		else:
			spans_pred = dict_bucket2span_pred[bucket_interval]




		confidence_low, confidence_up = compute_confidence_interval_f1_pos(spans_true, spans_pred, dict_span2sid, dict_span2sid_pred)

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
						error_entity_list.append(format4json_tc(dict_chunkid2span[span_true]) + "|||" + tag_true + "|||" + dict_pos2tag_pred[pos_true])
				else:
					#error_entity_list.append(format4json_tc(dict_chunkid2span[span_true]) + "|||" + tag_true + "|||" + "O")
					continue



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



def getBucketF1_chunk(dict_bucket2span, dict_bucket2span_pred, dict_span2sid, dict_span2sid_pred, dict_chunkid2span, dict_chunkid2span_pred):
	print('------------------ attribute')
	dict_bucket2f1 = {}




    # predict:  2_3 -> NER
	dict_pos2tag_pred = {}
	for k_bucket_eval, spans_pred in dict_bucket2span_pred.items():
		for span_pred in spans_pred:
			pos_pred = "_".join(span_pred.split("_")[0:2])
			tag_pred = span_pred.split("_")[-1]
			dict_pos2tag_pred[pos_pred] = tag_pred


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




def getErrorCase_semp(text_list, sql_true_list, sql_pred_list, is_match_list):
    errorCase_list = []
    for text, sql_true, sql_pred, is_match in zip(text_list, sql_true_list, sql_pred_list, is_match_list):
        if is_match == "0":
            errorCase_list.append(format4json_tc(text) + "|||" + format4json_tc(sql_true) + "|||" + format4json_tc(sql_pred) )
    return errorCase_list

def getBucketAcc_with_errorCase_semp(dict_bucket2span, dict_bucket2span_pred, dict_sid2sentpair):


    # The structure of span_true or span_pred
    # 2345|||Positive
    # 2345 represents sentence id
    # Positive represents the "label" of this instance

    dict_bucket2f1 = {}


    for bucket_interval, spans_true in dict_bucket2span.items():
        spans_pred = []


        # print('bucket_interval: ',bucket_interval)
        if bucket_interval not in dict_bucket2span_pred.keys():
            #print(bucket_interval)
            raise ValueError("Predict Label Bucketing Errors")
        else:
            spans_pred = dict_bucket2span_pred[bucket_interval]



        # loop over samples from a given bucket
        error_case_bucket_list = []
        for info_true, info_pred in zip(spans_true, spans_pred):
            sid_true, label_true = info_true.split("|||")
            sid_pred, label_pred = info_pred.split("|||")
            if sid_true != sid_pred:
                continue

            sent = dict_sid2sentpair[sid_true]
            if label_true != label_pred:
                error_case_info  =  sent
                error_case_bucket_list.append(error_case_info)


        accuracy_each_bucket = accuracy(spans_pred, spans_true)
        # print("debug: span_pred:\t")
        # print(spans_pred)
        confidence_low, confidence_up = compute_confidence_interval_acc(spans_pred, spans_true)
        dict_bucket2f1[bucket_interval] = [accuracy_each_bucket, len(spans_true), confidence_low, confidence_up, error_case_bucket_list]

        # print(error_case_bucket_list)

        print("accuracy_each_bucket:\t", accuracy_each_bucket)

    return sortDict(dict_bucket2f1)

