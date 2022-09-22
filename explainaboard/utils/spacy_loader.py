"""Loader for spacy models."""

from __future__ import annotations

import spacy
from spacy.language import Language


class SpacyLoader:
    """Loader for spacy models.

    This should be used in a singleton fashion to
    ensure that we don't load the same spacy model multiple times. It also
    encapsulates `spacy.load()` so we don't load big spacy models unless it's
    necessary.
    """

    _models: dict[str, Language] = {}

    def get_model(self, name: str) -> Language:
        """Loads a spacy model if it's not in memory and returns it.

        Args:
          name: name of the model, any valid identifier for `spacy.load()`
            should work, e.g. en_core_web_sm

        Returns:
          a spacy `Language` object
        """
        if name not in self._models:
            self._models[name] = spacy.load(name)
        return self._models[name]


# singleton spacy loader to keep one copy of each model in memory
spacy_loader = SpacyLoader()


def get_named_entities(text: str, system_name="en_core_web_sm") -> tuple[str]:
    """Use spacy to extract named entities from `text`.

    All other spacy components are disabled to improve speed.
    """
    return spacy_loader.get_model(system_name)(
        text, disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"]
    ).ents
