from __future__ import annotations

from functools import lru_cache


class Tokenizer:
    def __call__(self, text: str) -> list[str]:
        """
        Tokenize a string into a list of tokens
        :param text: The string to tokenize
        :returns: The list of tokens
        """
        raise NotImplementedError

    def detokenize(self, tokens: list[str]) -> str:
        """
        Detokenize a list of tokens into a string
        :param tokens: A list of tokens
        :returns: The detokenized string
        """
        raise NotImplementedError


class SingleSpaceTokenizer(Tokenizer):
    """
    Split a string on a single ascii space
    """

    @lru_cache(maxsize=20)
    def __call__(self, text: str) -> list[str]:
        return text.split(' ')

    def detokenize(self, tokens: list[str]) -> str:
        return ' '.join(tokens)
