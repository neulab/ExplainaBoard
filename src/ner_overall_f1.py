import codecs
import numpy as np



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

# def run_evaluate(self, sess, test, tags):
def evaluate(words,labels_pred, labels):
	"""
	labels_pred, labels, words: are sent-level list
	eg: words --> [[i love shanghai],[i love u],[i do not know]]
	words,pred, right: is a sequence, is label index or word index.
	Evaluates performance on test set

	"""
	# true_tags = ['PER', 'LOC', 'ORG', 'PERSON', 'person', 'loc', 'company']
	accs = []
	correct_preds, total_correct, total_preds = 0., 0., 0.

	for lab, lab_pred, word_sent in zip(labels, labels_pred, words):
		accs += [a == b for (a, b) in zip(lab, lab_pred)]
		lab_chunks = set(get_chunks(lab))
		lab_pred_chunks = set(get_chunks(lab_pred))
		correct_preds += len(lab_chunks & lab_pred_chunks)
		total_preds += len(lab_pred_chunks)
		total_correct += len(lab_chunks)

	p = correct_preds / total_preds if correct_preds > 0 else 0
	r = correct_preds / total_correct if correct_preds > 0 else 0
	f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
	acc = np.mean(accs)
	return acc, f1, p, r


def evaluate_each_class(words,labels_pred, labels,class_type):
	# class_type:PER or LOC or ORG
	index = 0

	accs = []
	correct_preds, total_correct, total_preds = 0., 0., 0.
	correct_preds_cla_type, total_preds_cla_type, total_correct_cla_type = 0., 0., 0.

	for lab, lab_pred, word_sent in zip(labels, labels_pred, words):
		lab_pre_class_type = []
		lab_class_type = []

		# accs += [a==b for (a, b) in zip(lab, lab_pred)]
		lab_chunks = get_chunks(lab)
		lab_pred_chunks = get_chunks(lab_pred)
		for i in range(len(lab_pred_chunks)):
			if lab_pred_chunks[i][0] == class_type:
				lab_pre_class_type.append(lab_pred_chunks[i])
		lab_pre_class_type_c = set(lab_pre_class_type)

		for i in range(len(lab_chunks)):
			if lab_chunks[i][0] == class_type:
				lab_class_type.append(lab_chunks[i])
		lab_class_type_c = set(lab_class_type)

		lab_chunksss = set(lab_chunks)
		correct_preds_cla_type += len(lab_pre_class_type_c & lab_chunksss)
		total_preds_cla_type += len(lab_pre_class_type_c)
		total_correct_cla_type += len(lab_class_type_c)

	p = correct_preds_cla_type / total_preds_cla_type if correct_preds_cla_type > 0 else 0
	r = correct_preds_cla_type / total_correct_cla_type if correct_preds_cla_type > 0 else 0
	f1 = 2 * p * r / (p + r) if correct_preds_cla_type > 0 else 0
	# acc = np.mean(accs)
	return f1, p, r

def evaluate_chunk_level(pred_chunks,true_chunks):
	# print(len(pred_chunks), len(true_chunks))
	# if len(pred_chunks) != len(true_chunks):
	# 	print("Error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!: len(pred_chunks) != len(true_chunks)")
	# 	exit()
	correct_preds, total_correct, total_preds = 0., 0., 0.
	correct_preds = len(set(true_chunks) & set(pred_chunks))
	total_preds = len(pred_chunks)
	total_correct = len(true_chunks)

	# print("****** debug *************")
	# print("correct_preds:\t", correct_preds)
	# print("total_preds:\t", total_preds)
	# print("total_correct:\t", total_correct)

	p = correct_preds / total_preds if correct_preds > 0 else 0
	r = correct_preds / total_correct if correct_preds > 0 else 0
	f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
	# acc = np.mean(accs)
	return f1, p, r

def evaluate_each_class_listone(words,labels_pred, labels,class_type):
	'''
	words,labels_pred, labels is list
	eg: labels  = [b-per, i-per,b-org,o,o,o, ...]
	:return:
	'''

	correct_preds, total_correct, total_preds = 0., 0., 0.
	correct_preds_cla_type, total_preds_cla_type, total_correct_cla_type = 0., 0., 0.

	lab_pre_class_type =[]
	lab_class_type =[]
	true_chunks = get_chunks(labels)
	pred_chunks = get_chunks(labels_pred)
	for i in range(len(pred_chunks)):
		if pred_chunks[i][0] == class_type:
			lab_pre_class_type.append(pred_chunks[i])
	lab_pre_class_type_c = set(lab_pre_class_type)

	for i in range(len(true_chunks)):
		if true_chunks[i][0] == class_type:
			lab_class_type.append(true_chunks[i])
	lab_class_type_c = set(lab_class_type)

	lab_chunksss = set(true_chunks)
	correct_preds_cla_type += len(lab_pre_class_type_c & lab_chunksss)
	total_preds_cla_type += len(lab_pre_class_type_c)
	total_correct_cla_type += len(lab_class_type_c)

	p = correct_preds_cla_type / total_preds_cla_type if correct_preds_cla_type > 0 else 0
	r = correct_preds_cla_type / total_correct_cla_type if correct_preds_cla_type > 0 else 0
	f1 = 2 * p * r / (p + r) if correct_preds_cla_type > 0 else 0
	# acc = np.mean(accs)
	return f1, p, r,len(lab_class_type)




# if __name__ == '__main__':
# 	max_sent = 10
# 	tags = {'0': 0,
# 			'B-PER': 1, 'I-PER': 2,
# 			'B-LOC': 3, 'I-LOC': 4,
# 			'B-ORG': 5, 'I-ORG': 6,
# 			'B-OTHER': 7, 'I-OTHER': 8,
# 			'O': 9}
# 	labels_pred = [
# 		[9, 9, 9, 1, 3, 1, 2, 2, 0, 0],
# 		[9, 9, 9, 1, 3, 1, 2, 0, 0, 0]
# 	]
# 	labels = [
# 		[9, 9, 9, 9, 3, 1, 2, 2, 0, 0],
# 		[9, 9, 9, 9, 3, 1, 2, 2, 0, 0]
# 	]
# 	words = [
# 		[0, 0, 0, 0, 0, 3, 6, 8, 5, 7],
# 		[0, 0, 0, 4, 5, 6, 7, 9, 1, 7]
# 	]
# 	id_to_vocb = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j'}
# 	class_type = 'PER'
# 	acc, f1, p, r = evaluate(labels_pred, labels, words, tags, max_sent, id_to_vocb)
# 	print acc, f1, p, r
# 	f1, p, r = evaluate_each_class(labels_pred, labels, words, tags, max_sent, id_to_vocb, class_type)
# 	print f1, p, r

