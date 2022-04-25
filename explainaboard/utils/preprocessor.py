from __future__ import annotations

import abc
import re
import string
import sys
import unicodedata

from explainaboard.utils.tokenizer import MLQAMixTokenizer, SingleSpaceTokenizer


@abc.abstractmethod
class Preprocessor:
    def __init__(self, language: str = "en"):
        self.language = language

    def set_language(self, language: str):
        self.language = language
        return self

    @abc.abstractmethod
    def __call__(self, text: str) -> str:
        """
        preprocess text
        :param text: text to be preprocessed
        :return: preprocessed text
        """
        ...


class QAPreprocessor(Preprocessor):
    """
    A preprocessor to process answers in extractive QA tasks.
    Currently it is based on the MLQA paper.
    """

    PUNCT = {
        chr(i)
        for i in range(sys.maxunicode)
        if unicodedata.category(chr(i)).startswith('P')
    }.union(string.punctuation)
    WHITESPACE_LANGS = ['en', 'es', 'hi', 'vi', 'de', 'ar']
    MIXED_SEGMENTATION_LANGS = ['zh']

    ss_tokenizer = SingleSpaceTokenizer()
    mlqa_tokenizer = MLQAMixTokenizer()

    def __init__(self, language: str):
        self.language: str = language

    def normalize_answer(self, s: str, language: str) -> str:
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(text, lang):
            if lang == 'en' or lang is None:
                return re.sub(r'\b(a|an|the)\b', ' ', text)
            elif lang == 'es':
                return re.sub(r'\b(un|una|unos|unas|el|la|los|las)\b', ' ', text)
            elif lang == 'hi':
                return text  # Hindi does not have formal articles
            elif lang == 'vi':
                return re.sub(r'\b(của|là|cái|chiếc|những)\b', ' ', text)
            elif lang == 'de':
                return re.sub(
                    r'\b(ein|eine|einen|einem|eines|einer|der|die|das|den|dem|'
                    r'des)\b',
                    ' ',
                    text,
                )
            elif lang == 'ar':
                # TODO(Pengfei): W605 invalid escape sequence '\s'
                return re.sub('\sال^|ال', ' ', text)  # noqa
            elif lang == 'zh':
                return text  # Chinese does not have formal articles
            else:  # TODO(Pengfei): is this too strong?
                raise Exception('Unknown Language {}'.format(lang))

        def white_space_fix(text, lang):
            if lang in self.WHITESPACE_LANGS or lang is None:
                tokens = self.ss_tokenizer(text)
            elif lang in self.MIXED_SEGMENTATION_LANGS:
                tokens = self.mlqa_tokenizer(text)
            else:
                raise Exception('Unknown Language {}'.format(lang))
            return ' '.join([t for t in tokens if t.strip() != ''])

        def remove_punc(text):
            return ''.join(ch for ch in text if ch not in self.PUNCT)

        def lower(text):
            return text.lower()

        return white_space_fix(
            remove_articles(remove_punc(lower(s)), language), language
        )

    def __call__(self, text: str) -> str:
        return self.normalize_answer(text, self.language)
