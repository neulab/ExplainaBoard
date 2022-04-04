from __future__ import annotations

import abc
from functools import lru_cache
import re
import string
import sys
import unicodedata


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


class SingleSpaceTokenizer(Tokenizer):
    """
    Split a string on a single ascii space
    """

    @lru_cache(maxsize=20)
    def __call__(self, text: str) -> list[str]:
        return text.split(' ')

    def detokenize(self, tokens: list[str]) -> str:
        return ' '.join(tokens)


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
