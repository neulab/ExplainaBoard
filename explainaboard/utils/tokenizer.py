from __future__ import annotations

import abc
from functools import lru_cache
import re
import string
import sys
import unicodedata

from sacrebleu.tokenizers.tokenizer_intl import TokenizerV14International
from sacrebleu.tokenizers.tokenizer_ja_mecab import TokenizerJaMecab
from sacrebleu.tokenizers.tokenizer_zh import TokenizerZh

from explainaboard import TaskType


def get_default_tokenizer(task_type: TaskType, lang: str) -> Tokenizer:
    cond_gen_tasks = {
        TaskType.conditional_generation,
        TaskType.machine_translation,
        TaskType.summarization,
    }
    if task_type in cond_gen_tasks:
        if lang == 'zh':
            return SacreBleuTokenizer(variety='zh')
        elif lang == 'ja':
            return SacreBleuTokenizer(variety='ja')
        else:
            return SacreBleuTokenizer(variety='intl')
    else:
        return SingleSpaceTokenizer()


class Tokenizer:
    @abc.abstractmethod
    def __call__(self, text: str) -> list[str]:
        """
        Tokenize a string into a list of tokens
        :param text: The string to tokenize
        :returns: The list of tokens
        """
        ...

    @abc.abstractmethod
    def detokenize(self, tokens: list[str]) -> str:
        """
        Detokenize a list of tokens into a string
        :param tokens: A list of tokens
        :returns: The detokenized string
        """
        ...

    @abc.abstractmethod
    def json_repr(self) -> dict:
        """
        Return a representation of this class that is serializable in json
        """
        ...


class SingleSpaceTokenizer(Tokenizer):
    """
    Split a string on a single ascii space
    """

    @lru_cache(maxsize=20)
    def __call__(self, text: str) -> list[str]:
        return text.split(' ')

    def detokenize(self, tokens: list[str]) -> str:
        return ' '.join(tokens)

    def json_repr(self):
        return {'cls': 'SingleSpaceTokenizer'}


class SacreBleuTokenizer(Tokenizer):
    """
    Split a string based on the strategy in SacreBLEU
    """

    def __init__(self, variety: str = 'intl'):
        self.variety = variety
        if variety == 'intl':
            self.tokenizer = TokenizerV14International()
        elif variety == 'zh':
            self.tokenizer = TokenizerZh()
        elif variety == 'ja-mecab':
            self.tokenizer = TokenizerJaMecab()
        else:
            raise ValueError(f'Illegal variety of SacreBleuTokenizer: {variety}')

    @lru_cache(maxsize=20)
    def __call__(self, text: str) -> list[str]:
        return self.tokenizer(text).split(' ')

    def detokenize(self, tokens: list[str]) -> str:
        raise NotImplementedError

    def json_repr(self):
        return {'cls': 'SacreBleuTokenizer', 'variety': self.variety}


class MLQAMixTokenizer(Tokenizer):

    ss_tokenizer = SingleSpaceTokenizer()
    PUNCT = {
        chr(i)
        for i in range(sys.maxunicode)
        if unicodedata.category(chr(i)).startswith('P')
    }.union(string.punctuation)

    @lru_cache(maxsize=20)
    def __call__(self, text: str) -> list[str]:

        segs_out = []
        temp_str = ""
        for char in text:
            if re.search(r'[\u4e00-\u9fa5]', char) or char in self.PUNCT:
                if temp_str != "":
                    ss = self.ss_tokenizer(temp_str)
                    segs_out.extend(ss)
                    temp_str = ""
                segs_out.append(char)
            else:
                temp_str += char

        if temp_str != "":
            ss = self.ss_tokenizer(temp_str)
            segs_out.extend(ss)

        return segs_out

    def detokenize(self, tokens: list[str]) -> str:
        raise NotImplementedError

    def json_repr(self):
        return {'cls': 'MLQAMixTokenizer'}
