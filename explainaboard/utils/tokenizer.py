"""Definition of tokenizers."""

from __future__ import annotations

import abc
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
import re
import string
import sys
from typing import final, overload
import unicodedata

from sacrebleu.tokenizers import BaseTokenizer
from sacrebleu.tokenizers.tokenizer_intl import TokenizerV14International
from sacrebleu.tokenizers.tokenizer_ja_mecab import TokenizerJaMecab
from sacrebleu.tokenizers.tokenizer_zh import TokenizerZh

from explainaboard import TaskType
from explainaboard.serialization.registry import TypeRegistry
from explainaboard.serialization.serializers import PrimitiveSerializer
from explainaboard.serialization.types import Serializable, SerializableData
from explainaboard.utils.typing_utils import narrow


def get_default_tokenizer(task_type: TaskType, lang: str | None) -> Tokenizer:
    cond_gen_tasks = {
        TaskType.conditional_generation,
        TaskType.machine_translation,
        TaskType.summarization,
    }
    if task_type in cond_gen_tasks or task_type == TaskType.nlg_meta_evaluation:
        if lang == 'zh':
            return SacreBleuTokenizer(variety='zh')
        elif lang == 'ja':
            return SacreBleuTokenizer(variety='ja-mecab')
        elif lang == 'python':
            return SacreBleuTokenizer(variety='conala')
        else:
            return SacreBleuTokenizer(variety='intl')
    else:
        return SingleSpaceTokenizer()


@final
@dataclass(frozen=True)
class TokenSeq(Sequence):
    """Dataclass representing a list of tokens and its original positions."""

    strs: list[str]
    positions: list[int]

    def __post_init__(self):
        if len(self.strs) != len(self.positions):
            raise ValueError("strs and positions must be the same length.")

    @overload
    def __getitem__(self, item: int) -> str:
        ...

    @overload
    def __getitem__(self, item: slice) -> Sequence[str]:
        ...

    def __getitem__(self, item: int | slice) -> str | Sequence[str]:
        return self.strs[item]

    def __len__(self) -> int:
        return len(self.strs)

    def __iter__(self):
        return iter(self.strs)

    @staticmethod
    def from_orig_and_tokens(orig: str, tokens: list[str]) -> TokenSeq:
        """Helper to generate TokenSeq from given string and tokens.

        Args:
            orig: Original text.
            tokens: List of tokens. Elements must be sorted by the same order in `orig`.

        Returns:
            TokenSeq constructed from given arguments.
        """
        start = 0
        strs: list[str] = []
        positions: list[int] = []

        for x in tokens:
            next_start = orig.find(x, start)
            if next_start == -1:
                raise ValueError(
                    "Could not find a token in the original text. "
                    f"orig={orig}, tokens={tokens}"
                )
            strs.append(x)
            positions.append(next_start)
            start = next_start

        return TokenSeq(strs, positions)


class Tokenizer(Serializable, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, text: str) -> TokenSeq:
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

    def serialize(self) -> dict[str, SerializableData]:
        """See Serializable.serialize."""
        return {}

    @classmethod
    def deserialize(cls, data: dict[str, SerializableData]) -> Serializable:
        """See Serializable.deserialize."""
        return cls()


_tokenizer_registry = TypeRegistry[Serializable]()


def get_tokenizer_serializer() -> PrimitiveSerializer:
    """Create a serializer for tokenizers.

    Returns:
        A serializer object for tokenizer classes.
    """
    return PrimitiveSerializer(_tokenizer_registry)


@final
@_tokenizer_registry.register("SingleSpaceTokenizer")
class SingleSpaceTokenizer(Tokenizer):
    """
    Split a string on a single ascii space
    """

    @lru_cache(maxsize=20)
    def __call__(self, text: str) -> TokenSeq:
        start = 0
        strs: list[str] = []
        positions: list[int] = []

        for x in text.split(' '):
            strs.append(x)
            positions.append(start)
            start = start + len(x) + 1

        return TokenSeq(strs, positions)

    def detokenize(self, tokens: list[str]) -> str:
        return ' '.join(tokens)


class TokenizerConala(BaseTokenizer):
    """
    A SacreBLEU style tokenizer of the tokenizer that we use for BLEU score over Python
    code, as used by the CoNaLa corpus.
    Originally from Wang Ling et al., Latent Predictor Networks for Code Generation
    """

    def __call__(self, text: str) -> str:
        """
        The tokenizer that we use for BLEU score over code, used by the CoNaLa corpus.
        Originally from Wang Ling et al., Latent Predictor Networks for Code Generation
        :param text: string containing a code snippet
        :return: space-separated tokens
        """
        text = re.sub(r'([^A-Za-z0-9_])', r' \1 ', text)
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.replace('"', '`')
        text = text.replace('\'', '`')
        text = text.strip(' ')

        return text


@final
@_tokenizer_registry.register("SacreBleuTokenizer")
class SacreBleuTokenizer(Tokenizer):
    """
    Split a string based on the strategy in SacreBLEU
    """

    @staticmethod
    def _get_normalizer(variety: str) -> Callable[[str], str]:
        """Helper to obtain a normalizer function associated to the variety.

        Args:
            variety: Name of the tokenization toolchain.

        Returns:
            Function to normalize texts.
        """
        if variety == 'conala':
            trdict = str.maketrans("'\"", "``")
            return lambda text: text.translate(trdict)
        else:
            return lambda text: text

    @staticmethod
    def _get_tokenizer(variety: str) -> Callable[[str], str]:
        """Helper to obtain an inner tokenizer function. associated to the variety.

        Args:
            variety: Name of the tokenization toolchain.

        Returns:
            Runction to tokenize texts to space-joined tokens.
        """
        if variety == 'intl':
            return TokenizerV14International()
        elif variety == 'zh':
            return TokenizerZh()
        elif variety == 'ja-mecab':
            return TokenizerJaMecab()
        elif variety == 'conala':
            return TokenizerConala()
        else:
            raise ValueError(f'Illegal variety of SacreBleuTokenizer: {variety}')

    def __init__(self, variety: str = 'intl'):
        self._variety = variety
        self._normalizer = self._get_normalizer(self._variety)
        self._tokenizer = self._get_tokenizer(self._variety)

    @lru_cache(maxsize=20)
    def __call__(self, text: str) -> TokenSeq:
        return TokenSeq.from_orig_and_tokens(
            self._normalizer(text), self._tokenizer(text).split(' ')
        )

    def detokenize(self, tokens: list[str]) -> str:
        raise NotImplementedError

    def serialize(self) -> dict[str, SerializableData]:
        return {"variety": self._variety}

    @classmethod
    def deserialize(cls, data: dict[str, SerializableData]) -> Serializable:
        return cls(variety=narrow(str, data["variety"]))


@final
@_tokenizer_registry.register("MLQAMixTokenizer")
class MLQAMixTokenizer(Tokenizer):
    def __init__(self) -> None:
        self._ss_tokenizer = SingleSpaceTokenizer()
        self._punct = {
            chr(i)
            for i in range(sys.maxunicode)
            if unicodedata.category(chr(i)).startswith('P')
        }.union(string.punctuation)

    @lru_cache(maxsize=20)
    def __call__(self, text: str) -> TokenSeq:

        segs_out: list[str] = []
        temp_str = ""
        for char in text:
            if re.search(r'[\u4e00-\u9fa5]', char) or char in self._punct:
                if temp_str != "":
                    ss = self._ss_tokenizer(temp_str)
                    segs_out.extend(ss)
                    temp_str = ""
                segs_out.append(char)
            else:
                temp_str += char

        if temp_str != "":
            ss = self._ss_tokenizer(temp_str)
            segs_out.extend(ss)

        return TokenSeq.from_orig_and_tokens(text, segs_out)

    def detokenize(self, tokens: list[str]) -> str:
        raise NotImplementedError
