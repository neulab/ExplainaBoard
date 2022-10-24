"""Preprocessing utilities.

This module provides several str -> str functions, which are used to preprocess input
texts in several tasks.
"""

from __future__ import annotations

import re
import string
import sys
import unicodedata

from explainaboard.utils.tokenizer import (
    MLQAMixTokenizer,
    SingleSpaceTokenizer,
    Tokenizer,
)


class MapPreprocessor:
    """Map a single string to another string based on a dictionary."""

    def __init__(self, dictionary: dict[str, str]):
        """Initializes MapPreprocessor.

        Args:
            dictionary: Dictionary to map a specific string to another.
        """
        self._dictionary = dictionary

    def __call__(self, text: str) -> str:
        """Map text s into another text based on the dictionary.

        If the string doesn't exist in the dictionary, return the original string.

        Args:
            text: Text to be processed by this preprocessor.

        Returns:
            Processed text.
        """
        return self._dictionary.get(text, text)


class ExtractiveQAPreprocessor:
    """Preprocessor to regularize answers in extractive QA tasks.

    Currently the implementation is based on the MLQA paper.
    """

    _PUNCT = {
        chr(i)
        for i in range(sys.maxunicode)
        if unicodedata.category(chr(i)).startswith("P")
    }.union(string.punctuation)
    _MIXED_SEGMENTATION_LANGS = ["zh"]

    def __init__(self, language: str | None) -> None:
        """Initialises ExtractiveQAPreprocessor.

        Args:
            language: The language code of texts, or None for unspecified languages.
        """
        if language in ["en", "eng"]:
            pattern = r"\b(a|an|the)\b"
        elif language in ["es", "spa"]:
            pattern = r"\b(un|una|unos|unas|el|la|los|las)\b"
        elif language in ["vi", "vie"]:
            pattern = r"\b(của|là|cái|chiếc|những)\b"
        elif language in ["de", "deu"]:
            pattern = r"\b(ein|eine|einen|einem|eines|einer|der|die|das|den|dem|des)\b"
        elif language in ["ar", "ara"]:
            pattern = r"\sال^|ال"
        else:
            pattern = None

        self._article_sanitizer = re.compile(pattern) if pattern is not None else None

        if language in self._MIXED_SEGMENTATION_LANGS:
            self._tokenizer: Tokenizer = MLQAMixTokenizer()
        else:
            self._tokenizer = SingleSpaceTokenizer()

    def _remove_articles(self, text: str) -> str:
        return (
            self._article_sanitizer.sub(" ", text)
            if self._article_sanitizer is not None
            else text
        )

    def _white_space_fix(self, text: str) -> str:
        return " ".join(t for t in self._tokenizer(text) if t.strip() != "")

    def _remove_punc(self, text: str) -> str:
        return "".join(ch for ch in text if ch not in self._PUNCT)

    def __call__(self, text: str) -> str:
        """Process texts.

        Args:
            text: Text to be processed by this preprocessor.

        Returns:
            Processed text.
        """
        return self._white_space_fix(
            self._remove_articles(self._remove_punc(text.lower()))
        )
