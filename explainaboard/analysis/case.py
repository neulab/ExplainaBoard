from __future__ import annotations

import dataclasses
from dataclasses import dataclass


@dataclass
class AnalysisCase:
    """
    A class to represent cases to display to users for analysis.
    :param sample_id: The ID of a single sample
    """

    sample_id: int
    features: dict

    def __post_init__(self):
        if isinstance(self.sample_id, str):
            raise ValueError

    @classmethod
    def dict_conv(cls, k: str, v: dict):
        """
        A deserialization utility function that takes in a key corresponding to a
        parameter name, and dictionary corresponding to a serialized version of that
        parameter's value, then return the deserialized version of the value.
        :param k: the parameter name
        :param v: the parameter's value
        """
        return v

    @classmethod
    def from_dict(cls, data_dict: dict):
        field_names = set(f.name for f in dataclasses.fields(cls))
        return cls(
            **{k: cls.dict_conv(k, v) for k, v in data_dict.items() if k in field_names}
        )


@dataclass
class AnalysisCaseSpan(AnalysisCase):
    """
    A bucket case that highlights a span in text.
    :param text: The text that should be highlighted
    :param token_span: The span of tokens to be highlighted
    :param char_span: The span of characters to be highlighted
    :param location: The name of the feature (e.g. "text", "source", "reference") over
      which this span is calculated
    """

    token_span: tuple[int, int]
    char_span: tuple[int, int]
    text: str
    orig_str: str

    def __post_init__(self):
        if isinstance(self.token_span, str) or isinstance(self.char_span, str):
            raise ValueError


@dataclass
class AnalysisCaseMultiSpan(AnalysisCase):
    """
    A bucket case that highlights multiple spans in text
    :param spans: The spans that are highlighted
    """

    spans: list[AnalysisCaseSpan]


@dataclass
class AnalysisCaseLabeledSpan(AnalysisCaseSpan):
    """
    A bucket case that highlights a span in text along with a label.
    :param true_label: The actual label
    :param predicted_label: The label that is predicted
    """

    true_label: str
    predicted_label: str


@dataclass
class AnalysisCaseCollection:
    # This tuple is either tuple[float,float] (for continuous values) or tuple[str] for
    # discrete values
    # TODO(gneubig): add actual type annotation to this effect. at the moment it's a
    #    bit complicated due to the implementation of the bucketing functions
    interval: tuple
    samples: list[int]

    def __len__(self):
        return len(self.samples)
