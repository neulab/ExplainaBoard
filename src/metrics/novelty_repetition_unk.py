from nltk.corpus import stopwords, wordnet
from nltk import word_tokenize, pos_tag, sent_tokenize


stopwords = stopwords.words('english')
stop_list = [',', '.', '?', '!', '\'', '`', '\'s', ':', '-', '-lrb-', '-rrb-', '#', '--', '\'\'', '``', 'n\'t', '$',
             '(', ')']
for w in stop_list:
    stopwords.append(w)


def get_ngram(sentences, n):
    dic = {}
    if n == -1:
        for sent in sentences:
            if sent in dic:
                dic[sent] += 1
            else:
                dic[sent] = 1
        return dic

    for sent in sentences:
        tokens = word_tokenize(sent)
        # if n == 1:
        tokens = [token for token in tokens if token not in stopwords]
        s_len = len(tokens)
        for k in range(s_len):
            if k + n > s_len:
                break
            update_key = ' '.join(tokens[k: k + n])
            if update_key in dic:
                dic[update_key] += 1
            else:
                dic[update_key] = 1
    return dic


def repetition(summary_sentences, ngram):
    '''
    input one sample of summary_sentences calculate the repetition
    :param summary_sentences: list of summary sentences, 2-D, every item of the list is a list of sentences for the summary of one sample
    :param ngram:
    :return:
    '''
    rate = []

    for summary in summary_sentences:
        dic = get_ngram(summary, ngram)
        cnt_all = 0
        cnt_rep = 0
        for key, value in dic.items():
            cnt_all += value
            if value > 2:
                cnt_rep += value - 1
        if not cnt_all == 0:
            rate.append(cnt_rep / cnt_all)
    return rate


def repetition_oneSample(summary, ngram):

    dic = get_ngram(summary, ngram)
    cnt_all = 0
    cnt_rep = 0
    rate = 0
    for key, value in dic.items():
        # print(key)
        cnt_all += value
        if value > 2:
            cnt_rep += value - 1
    if not cnt_all == 0:
        rate = cnt_rep / cnt_all
    return rate


def novelty_oneSample(doc, summary, ngram):
    rate = 0
    score = 0
    all_cnt = 0
    pred_dic = get_ngram(summary, ngram)
    raw_keys = get_ngram(doc, ngram).keys()
    for key, value in pred_dic.items():
        all_cnt += value
        if not key in raw_keys:
            score += value
    if not all_cnt == 0:
        rate = (score / all_cnt)
    return rate


def novelty(source_sentences, summary_sentences, ngram):
    '''
    input one sample of summary_sentences calculate the novelty
    :param source_sentences:list of source sentences, 2-D, every item of the list is a list of sentences for the source of one sample
    :param summary_sentences:list of summary sentences, 2-D, every item of the list is a list of sentences for the summary of one sample
    :param ngram:
    :return:
    '''

    raws = source_sentences
    preds = summary_sentences

    assert len(preds) == len(raws), "preds and raws have inequal number of samples!"
    rate = []
    for i in range(len(preds)):
        score = 0
        all_cnt = 0
        pred_dic = get_ngram(preds[i], ngram)
        raw_keys = get_ngram(raws[i], ngram).keys()
        for key, value in pred_dic.items():
            all_cnt += value
            if not key in raw_keys:
                score += value
        if not all_cnt == 0:
            rate.append(score / all_cnt)
    return rate



