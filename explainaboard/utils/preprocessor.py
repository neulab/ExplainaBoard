"""Preprocessing utilities."""

from __future__ import annotations

import abc
import re
import string
import sys
from typing import Any, Optional
import unicodedata

from explainaboard.utils.tokenizer import MLQAMixTokenizer, SingleSpaceTokenizer


@abc.abstractmethod
class Preprocessor:
    """A preprocesor that applies processing to strings."""

    def __init__(
        self, language: str | None = None, resources: Optional[dict[str, Any]] = None
    ):
        """Constructor.

        Args:
            language: The language of the incoming text
            resources: Any resources necessary to do the preprocessing
        """
        self.language = language
        self.resources = resources or self.default_resources()

    def set_language(self, language: str) -> Preprocessor:
        """Set the language of the preprocessor.

        Args:
            language: The language to set

        Returns:
            The preprocessor itself.
        """
        self.language = language
        return self

    def default_resources(self) -> dict[str, Any]:
        """Returns default features for this processor."""
        return {}

    @abc.abstractmethod
    def process(self, s: str, resources: dict[str, Any]) -> str:
        """The processing function that applies some sort of processing.

        Args:
            s: The string to process.
            resources: The resources to use in processing.

        Returns:
            The processed string.
        """
        ...

    def __call__(self, text: str) -> str:
        """Preprocess text.

        Args:
            text: text to be preprocessed

        Returns:
            preprocessed text
        """
        return self.process(text, self.resources)


class MapPreprocessor(Preprocessor):
    """Map a single string to another string based on a dictionary."""

    def default_resources(self) -> dict:
        """Returns default features for this processor."""
        return {"dictionary": {}}

    def process(self, s: str, resources: dict[str, Any]) -> str:
        """Map text s into another text based on the dictionary.

        If the string doesn't exist in the dictionary, return the original string.

        Args:
            s: The string to process.
            resources: The resources to use in processing.

        Returns:
            The processed string.
        """
        return resources['dictionary'].get(s, s)


class KGMapPreprocessor(Preprocessor):
    """A mapping preprocessor specifically used in the KG link prediction processor."""

    def default_resources(self) -> dict:
        """Returns default features for this processor."""
        return {"dictionary": {}}

    def process(self, s: str, resources: dict[str, Any]) -> str:
        """Map text s into another text based on the dictionary.

        If the string doesn't exist in the dictionary, return the original string.

        Args:
            s: The string to process.
            resources: The resources to use in processing.

        Returns:
            The processed string.
        """
        return (
            resources['dictionary'][s]["label"] if s in resources['dictionary'] else s
        )


class ExtractiveQAPreprocessor(Preprocessor):
    """A preprocessor to process answers in extractive QA tasks.

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

    def default_resources(self) -> dict:
        """Returns default features for this processor."""
        return {"language": self.language}

    def _remove_articles(self, text: str, lang: str) -> str:
        if lang in ['en', 'eng']:
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        elif lang in ['es', 'spa']:
            return re.sub(r'\b(un|una|unos|unas|el|la|los|las)\b', ' ', text)
        elif lang in ['vi', 'vie']:
            return re.sub(r'\b(của|là|cái|chiếc|những)\b', ' ', text)
        elif lang in ['de', 'deu']:
            return re.sub(
                r'\b(ein|eine|einen|einem|eines|einer|der|die|das|den|dem|' r'des)\b',
                ' ',
                text,
            )
        elif lang in ['ar', 'ara']:
            # TODO(Pengfei): W605 invalid escape sequence '\s'
            return re.sub('\sال^|ال', ' ', text)  # noqa
        else:
            return text

    def _white_space_fix(self, text: str, lang: str) -> str:

        if lang in self.MIXED_SEGMENTATION_LANGS:
            tokens = self.mlqa_tokenizer(text)
        else:
            tokens = self.ss_tokenizer(text)
        return ' '.join([t for t in tokens if t.strip() != ''])

    def _remove_punc(self, text: str) -> str:
        return ''.join(ch for ch in text if ch not in self.PUNCT)

    def process(self, s: str, resources: dict[str, Any]) -> str:
        """Lowercase text and remove punctuation, articles and extra whitespace."""
        language = resources['language']

        return self._white_space_fix(
            self._remove_articles(self._remove_punc(s.lower()), language), language
        )
