from collections import Counter, namedtuple
from functools import lru_cache

import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.util import ngrams

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    # TODO(odashi): Avoid programatic download: it requires unnecessary outbound
    # connection and won't work in offline systems.
    nltk.download('punkt')


class SUMAttribute:
    """
    We calculate the following attributes given a sample. They are all reference-free.
    * source_len
    * hypothesis_len
    * density
    * coverage
    * compression
    * repetition
    * novelty
    * copy_len
    """

    # TODO(odashi): Use dataclass instead.
    Match = namedtuple("Match", ("summary", "text", "length"))

    def __call__(self, texts, summaries):
        """texts: a list of source documents.
        summaries: a list of generated summaries.
        :return: a list of dics
        """
        out = []
        for text, summary in zip(texts, summaries):
            out.append(self.cal_attributes_each(text, summary))
        return out

    def get_schema(self):
        return {
            "attr_density": 0.0,
            "attr_coverage": 0.0,
            "attr_compression": 0.0,
            "attr_repetition": 0.0,
            "attr_novelty": 0.0,
            "attr_copy_len": 0.0,
            "attr_source_len": 0.0,
            "attr_hypothesis_len": 0.0,
        }

    @lru_cache(maxsize=10)
    def cal_attributes_each(self, text, summary):

        # Normalize text
        tokenized_text = word_tokenize(text)
        tokenized_summary = word_tokenize(summary)
        normalized_text = [str(t).lower() for t in tokenized_text]
        normalized_summary = [str(t).lower() for t in tokenized_summary]

        # Calculate matches
        matches = self.overlap(normalized_summary, normalized_text)
        summary_len = len(tokenized_summary)

        if summary_len == 0:
            density, coverage, compression = 0, 0, 0
        else:
            # Density
            density = sum(o.length**2 for o in matches) / summary_len
            # Coverage
            coverage = sum(o.length for o in matches) / summary_len
            # Compression
            compression = len(tokenized_text) / summary_len

        # Repetition
        repetition = self.cal_repetition(summary)
        # Novelty
        novelty = self.cal_novelty(text, summary)

        # Copy length
        copy_lens = [o.length for o in matches]
        if len(copy_lens) == 0:
            copy_len = 0
        else:
            copy_len = sum(copy_lens) / len(copy_lens)
        return {
            "attr_density": density,
            "attr_coverage": coverage,
            "attr_compression": compression,
            "attr_repetition": repetition,
            "attr_novelty": novelty,
            "attr_copy_len": copy_len,
            "attr_source_len": len(normalized_text),
            "attr_hypothesis_len": len(normalized_summary),
        }

    def get_ngrams(self, doc, n):
        doc = doc.lower()
        doc_sents = sent_tokenize(doc)
        _ngrams = []
        for sent in doc_sents:
            sent = word_tokenize(sent)
            _ngrams.extend(list(ngrams(sent, n=n)))
        return _ngrams

    def cal_novelty(self, text, summary, n=2):
        """Returns the novelty score.
        Novelty is the proportion of segments in the summaries that haven’t appeared in
        source documents. The segments can be instantiated as n-grams.
        """
        cnt_all = 0
        cnt_nov = 0
        _ngrams_text = self.get_ngrams(text, n=n)
        _ngrams_summary = self.get_ngrams(summary, n=n)
        counter_text = Counter(_ngrams_text)
        counter_summary = Counter(_ngrams_summary)
        for k, v in counter_summary.items():
            cnt_all += v
            if k not in counter_text:
                cnt_nov += v
        if cnt_all == 0:
            return 0
        else:
            return cnt_nov / cnt_all

    def cal_repetition(self, summary, n=3):
        """Measures the rate of repeated segments in summaries.
        We choose n-gram as segment unit.
        """
        cnt_all = 0
        cnt_rep = 0
        _ngrams = self.get_ngrams(summary, n=n)
        counter = Counter(_ngrams)
        for k, v in counter.items():
            cnt_all += v
            if v >= 2:
                cnt_rep += v - 1
        if cnt_all == 0:
            return 0
        else:
            return cnt_rep / cnt_all

    def overlap(self, a, b):
        """
        Return a list of Match objects between summary and text.
        This is a list of named tuples of the form (summary, text, length):
            - summary (int): the start index of the match in the summary
            - text (int): the start index of the match in the reference
            - length (int): the length of the extractive fragment
        """
        matches = []
        a_start = 0
        b_start = 0
        while a_start < len(a):
            best_match = None
            best_match_length = 0
            while b_start < len(b):
                if a[a_start] == b[b_start]:
                    a_end = a_start
                    b_end = b_start
                    while a_end < len(a) and b_end < len(b) and b[b_end] == a[a_end]:
                        b_end += 1
                        a_end += 1
                    length = a_end - a_start
                    if length > best_match_length:
                        best_match = SUMAttribute.Match(a_start, b_start, length)
                        best_match_length = length
                    b_start = b_end
                else:
                    b_start += 1
            b_start = 0
            if best_match:
                if best_match_length > 0:
                    matches.append(best_match)
                a_start += best_match_length
            else:
                a_start += 1
        return matches
