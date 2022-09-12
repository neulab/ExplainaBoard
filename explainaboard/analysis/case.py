"""Classes to express analysis cases."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Optional


@dataclass
class AnalysisCase:
    """A class to represent cases to display to users for analysis.

    Attributes:
        sample_id: The ID of a single sample
        features: A dictionary of features associated with the case
    """

    sample_id: int
    features: dict

    @classmethod
    def dict_conv(cls, k: str, v: dict):
        """A deserialization utility function.

        It takes in a key corresponding to a
        parameter name, and dictionary corresponding to a serialized version of that
        parameter's value, then return the deserialized version of the value.

        Args:
            k: the parameter name
            v: the parameter's value

        Returns:
            The value itself
        """
        return v

    @classmethod
    def from_dict(cls, data_dict: dict):
        """Deserialization function."""
        field_names = set(f.name for f in dataclasses.fields(cls))
        return cls(
            **{k: cls.dict_conv(k, v) for k, v in data_dict.items() if k in field_names}
        )


@dataclass
class AnalysisCaseSpan(AnalysisCase):
    """A bucket case that highlights a span in text.

    Attributes:
        token_span: The span of tokens to be highlighted
        char_span: The span of characters to be highlighted
        text: The text that should be highlighted
        location: The name of the feature (e.g. "text", "source", "reference") over
          which this span is calculated
    """

    token_span: tuple[int, int]
    char_span: tuple[int, int]
    text: str
    orig_str: str


@dataclass
class AnalysisCaseMultiSpan(AnalysisCase):
    """A bucket case that highlights multiple spans in text.

    Attributes:
        spans: The spans that are highlighted
    """

    spans: list[AnalysisCaseSpan]


@dataclass
class AnalysisCaseLabeledSpan(AnalysisCaseSpan):
    """A bucket case that highlights a span in text along with a label.

    Attributes:
        true_label: The actual label
        predicted_label: The label that is predicted
    """

    true_label: str
    predicted_label: str


@dataclass
class AnalysisCaseLabeledArgumentPair(AnalysisCase):
    """A bucket case that annotates a text block (pair of arguments).

    This is specifically designed for argument pair extraction task.

    Attributes:
        text: the text
        orig_str: the original string
        true_label: the true label of the output
        predicted_abel: the predicted label
        block_review_sentences: the number of review sentence
        block_review_tokens: the number of review tokens
        block_review_position: the relative position of review block
        block_reply_sentences: the number of reply sentence
        block_reply_tokens: the number of reply tokens
        block_reply_position: the relative position of reply block
    """

    text: str
    orig_str: str

    true_label: str
    predicted_label: str

    block_review_sentences: Optional[float] = None
    block_review_tokens: Optional[float] = None
    block_review_position: Optional[float] = None
    block_reply_sentences: Optional[float] = None
    block_reply_tokens: Optional[float] = None
    block_reply_position: Optional[float] = None


@dataclass
class AnalysisCaseCollection:
    """A collection of analysis cases.

    Attributes:
        samples: a list of integer IDs indexing back to the analysis cases in the level.
        interval: the bucket interval that this collection may refer to
        name: the name that the collection may refer to
    """

    samples: list[int]
    interval: tuple[float, float] | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        """Check that the input is valid."""
        if self.interval is None and self.name is None:
            raise ValueError("Either interval or name must have a value.")
        if self.interval is not None and self.name is not None:
            raise ValueError(
                "Both interval and name must not have values at the same time."
            )

    def __len__(self) -> int:
        """Return the size of the samples in the bucket."""
        return len(self.samples)
