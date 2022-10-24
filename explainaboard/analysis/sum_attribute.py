"""A class to calculate features for summarization."""

from __future__ import annotations

from collections import Counter, namedtuple
from functools import lru_cache

import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.util import ngrams

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    # TODO(odashi): Avoid programatic download: it requires unnecessary outbound
    # connection and won't work in offline systems.
    nltk.download("punkt")


class SUMAttribute:
    """This class calculates several attributes given a sample summary.

    These attributes are all refernce free.
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

    def __call__(self, texts: list[str], summaries: list[str]) -> list[dict]:
        """Calculate attributes of each pair of text and summary.

        Args:
            texts: a list of source documents.
            summaries: a list of generated summaries.

        Returns:
            A list of dicts with attributes.
        """
        out = []
        for text, summary in zip(texts, summaries):
            out.append(self.cal_attributes_each(text, summary))
        return out

    @lru_cache(maxsize=10)
    def cal_attributes_each(self, text: str, summary: str) -> dict[str, int | float]:
        """For a single instance, calculate the attributes of each text/summary pair.

        Args:
            text: The text.
            summary: The summary.

        Returns:
            Returns the summary.
        """
        # Normalize text
        tokenized_text = word_tokenize(text)
        tokenized_summary = word_tokenize(summary)
        normalized_text = [str(t).lower() for t in tokenized_text]
        normalized_summary = [str(t).lower() for t in tokenized_summary]

        # Calculate matches
        matches = self.overlap(normalized_summary, normalized_text)
        summary_len = len(tokenized_summary)

        if summary_len == 0:
            density, coverage, compression = 0.0, 0.0, 0.0
        else:
            # Density
            density = sum(float(o.length) ** 2 for o in matches) / summary_len
            # Coverage
            coverage = sum(float(o.length) for o in matches) / summary_len
            # Compression
            compression = float(len(tokenized_text)) / summary_len

        # Repetition
        repetition = self.cal_repetition(summary)
        # Novelty
        novelty = self.cal_novelty(text, summary)

        # Copy length
        copy_lens = [o.length for o in matches]
        if len(copy_lens) == 0:
            copy_len = 0.0
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

    def _get_ngrams(self, doc, n):
        doc = doc.lower()
        doc_sents = sent_tokenize(doc)
        _ngrams = []
        for sent in doc_sents:
            sent = word_tokenize(sent)
            _ngrams.extend(list(ngrams(sent, n=n)))
        return _ngrams

    def cal_novelty(self, text: str, summary: str, n: int = 2) -> float:
        """Returns the novelty score.

        Novelty is the proportion of segments in the summaries that haven’t appeared in
        source documents. The segments can be instantiated as n-grams.

        Args:
            text: The text.
            summary: The summary.
            n: The order of n-grams used in novelty calculation.

        Returns:
            The ratio of novel n-grams in the summary.
        """
        cnt_all = 0
        cnt_nov = 0
        _ngrams_text = self._get_ngrams(text, n=n)
        _ngrams_summary = self._get_ngrams(summary, n=n)
        counter_text: Counter = Counter(_ngrams_text)
        counter_summary: Counter = Counter(_ngrams_summary)
        for k, v in counter_summary.items():
            cnt_all += v
            if k not in counter_text:
                cnt_nov += v
        if cnt_all == 0:
            return 0
        else:
            return cnt_nov / cnt_all

    def cal_repetition(self, summary: str, n: int = 3) -> float:
        """Return the ratio of repeated segments in the summary.

        Args:
            summary: The summary.
            n: The length of the n-grams to be used in the calculation.

        Returns:
            The number of n-grams that are repeated in the summary.
        """
        cnt_all = 0
        cnt_rep = 0
        _ngrams = self._get_ngrams(summary, n=n)
        counter: Counter = Counter(_ngrams)
        for k, v in counter.items():
            cnt_all += v
            if v >= 2:
                cnt_rep += v - 1
        if cnt_all == 0:
            return 0
        else:
            return cnt_rep / cnt_all

    def overlap(self, summary: list[str], text: list[str]) -> list[Match]:
        """Return a list of Match objects between summary and text.

        This is a list of named tuples of the form (summary, text, length):
            - summary (int): the start index of the match in the summary
            - text (int): the start index of the match in the reference
            - length (int): the length of the extractive fragment

        Args:
            summary: the summary
            text: the text

        Returns:
            A list of Match objects indicating matches between the summary and text.
        """
        matches = []
        summary_start = 0
        text_start = 0
        while summary_start < len(summary):
            best_match = None
            best_match_length = 0
            while text_start < len(text):
                if summary[summary_start] == text[text_start]:
                    summary_end = summary_start
                    text_end = text_start
                    while (
                        summary_end < len(summary)
                        and text_end < len(text)
                        and text[text_end] == summary[summary_end]
                    ):
                        text_end += 1
                        summary_end += 1
                    length = summary_end - summary_start
                    if length > best_match_length:
                        best_match = SUMAttribute.Match(
                            summary_start, text_start, length
                        )
                        best_match_length = length
                    text_start = text_end
                else:
                    text_start += 1
            text_start = 0
            if best_match:
                if best_match_length > 0:
                    matches.append(best_match)
                summary_start += best_match_length
            else:
                summary_start += 1
        return matches
