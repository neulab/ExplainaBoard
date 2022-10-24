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

from explainaboard.serialization import common_registry
from explainaboard.serialization.types import Serializable, SerializableData
from explainaboard.utils.language_utils import (
    is_chinese_lang_code,
    is_japanese_lang_code,
)
from explainaboard.utils.typing_utils import narrow


def get_default_tokenizer(lang: str | None) -> Tokenizer:
    """Get the default tokenizer by language.

    Args:
        lang: The language

    Returns:
        A tokenizer
    """
    if is_chinese_lang_code(lang):
        return SacreBleuTokenizer(variety="zh")
    elif is_japanese_lang_code(lang):
        return SacreBleuTokenizer(variety="ja-mecab")
    elif lang == "python":
        return SacreBleuTokenizer(variety="conala")
    else:
        return SingleSpaceTokenizer()


@final
@dataclass(frozen=True)
class TokenSeq:
    """Dataclass representing a list of tokens and its original positions."""

    strs: list[str]
    positions: list[int]

    def __post_init__(self):
        """Perform checks of validity."""
        if len(self.strs) != len(self.positions):
            raise ValueError("strs and positions must be the same length.")

    @overload
    def __getitem__(self, index: int) -> str:  # noqa: D105: suppress bug
        ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[str]:  # noqa: D105: suppress bug
        ...

    def __getitem__(self, item: int | slice) -> str | Sequence[str]:
        """Get an item or slice from the sequence."""
        return self.strs[item]

    def __len__(self) -> int:
        """Get the length of the string sequence."""
        return len(self.strs)

    def __iter__(self):
        """Get an iterator over the strings."""
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
    """A class representing tokenization methods."""

    @abc.abstractmethod
    def __call__(self, text: str) -> TokenSeq:
        """Tokenize a string into a list of tokens.

        Args:
            text: The string to tokenize

        Returns:
            The list of tokens
        """
        ...

    @abc.abstractmethod
    def detokenize(self, tokens: list[str]) -> str:
        """Detokenize a list of tokens into a string.

        Args:
            tokens: A list of tokens

        Returns:
            The detokenized string
        """
        ...

    def serialize(self) -> dict[str, SerializableData]:
        """See Serializable.serialize."""
        return {}

    @classmethod
    def deserialize(cls, data: dict[str, SerializableData]) -> Serializable:
        """See Serializable.deserialize."""
        return cls()


@final
@common_registry.register("SingleSpaceTokenizer")
class SingleSpaceTokenizer(Tokenizer):
    """Split a string on a single ascii space."""

    @lru_cache(maxsize=20)
    def __call__(self, text: str) -> TokenSeq:
        """Perform tokenization.

        Args:
            text: The string to tokenize

        Returns:
            The tokenized sequence
        """
        start = 0
        strs: list[str] = []
        positions: list[int] = []

        for x in text.split(" "):
            strs.append(x)
            positions.append(start)
            start = start + len(x) + 1

        return TokenSeq(strs, positions)

    def detokenize(self, tokens: list[str]) -> str:
        """Detokenize the string by joining the strings together with white space.

        Args:
            tokens: The tokens to merge together.

        Returns:
            The detokenized strings.
        """
        return " ".join(tokens)


class TokenizerConala(BaseTokenizer):
    """A SacreBLEU style tokenizer for BLEU score over Python code.

    This is as used by the CoNaLa corpus.
    Originally from Wang Ling et al., Latent Predictor Networks for Code Generation
    """

    def __call__(self, text: str) -> str:
        """The tokenizer that we use for BLEU score over code.

        This is as used by the CoNaLa corpus.
        Originally from Wang Ling et al., Latent Predictor Networks for Code Generation

        Args:
            text: string containing a code snippet

        Returns:
            space-separated tokens
        """
        text = re.sub(r"([^A-Za-z0-9_])", r" \1 ", text)
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
        text = re.sub(r"\s+", " ", text)
        text = text.replace('"', "`")
        text = text.replace("'", "`")
        text = text.strip(" ")

        return text


@final
@common_registry.register("SacreBleuTokenizer")
class SacreBleuTokenizer(Tokenizer):
    """Split a string based on the strategy in SacreBLEU."""

    @staticmethod
    def _get_normalizer(variety: str) -> Callable[[str], str]:
        """Helper to obtain a normalizer function associated to the variety.

        Args:
            variety: Name of the tokenization toolchain.

        Returns:
            Function to normalize texts.
        """
        if variety == "conala":
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
        if variety == "intl":
            return TokenizerV14International()
        elif variety == "zh":
            return TokenizerZh()
        elif variety == "ja-mecab":
            return TokenizerJaMecab()
        elif variety == "conala":
            return TokenizerConala()
        else:
            raise ValueError(f"Illegal variety of SacreBleuTokenizer: {variety}")

    def __init__(self, variety: str = "intl"):
        """Constructor function.

        Args:
            variety: What variety of tokenizer to create, matching SacreBLEU.
        """
        self._variety = variety
        self._normalizer = self._get_normalizer(self._variety)
        self._tokenizer = self._get_tokenizer(self._variety)

    @lru_cache(maxsize=20)
    def __call__(self, text: str) -> TokenSeq:
        """Perform tokenization.

        Args:
            text: The string to tokenize

        Returns:
            The tokenized sequence
        """
        return TokenSeq.from_orig_and_tokens(
            self._normalizer(text), self._tokenizer(text).split(" ")
        )

    def detokenize(self, tokens: list[str]) -> str:
        """Detokenization (not implemented for this tokenizer)."""
        raise NotImplementedError

    def serialize(self) -> dict[str, SerializableData]:
        """Serialize all information about the tokenizer."""
        return {"variety": self._variety}

    @classmethod
    def deserialize(cls, data: dict[str, SerializableData]) -> Serializable:
        """Create a tokenizer from serialized information."""
        return cls(variety=narrow(str, data["variety"]))


@final
@common_registry.register("MLQAMixTokenizer")
class MLQAMixTokenizer(Tokenizer):
    """A tokenizers that is used for QA, based on the MLQA corpus."""

    def __init__(self) -> None:
        """Constructor."""
        self._ss_tokenizer = SingleSpaceTokenizer()
        self._punct = {
            chr(i)
            for i in range(sys.maxunicode)
            if unicodedata.category(chr(i)).startswith("P")
        }.union(string.punctuation)

    @lru_cache(maxsize=20)
    def __call__(self, text: str) -> TokenSeq:
        """Perform tokenization.

        Args:
            text: The string to tokenize

        Returns:
            The tokenized sequence
        """
        segs_out: list[str] = []
        temp_str = ""
        for char in text:
            if re.search(r"[\u4e00-\u9fa5]", char) or char in self._punct:
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
        """Detokenization (not implemented for this Tokenizer)."""
        raise NotImplementedError
