from __future__ import annotations

import abc
from typing import Any, Optional


def cap_feature(s):
    """
    Capitalization feature:
    0 = low caps
    1 = all caps
    2 = first letter caps
    3 = one capital (not first letter)
    """
    if s.lower() == s:
        return "low_caps"
    elif s.upper() == s:
        return "full_caps"
    elif s[0].upper() == s[0]:
        return "first_caps"
    else:
        return "not_first_caps"


def get_spans_from_bio(bio_seq: list[str]) -> list[tuple[str, int, int]]:
    """
    Takes in a BIO-tagged sequence of tokens, and returns tagged spans.
    :param bio_seq: A sequence of bio-tagged strings
                    such as ['O', 'B-PER', 'I-PER', 'B-ORG', 'O']
    :return: A sequence of spans in format (tag,begin,end),
             such as [('PER',1,3), ('ORG',3,4)]
    """

    default = 'O'
    # idx_to_tag = {idx: tag for tag, idx in tags.items()}
    spans = []
    span_type, span_start = None, -1
    for i, tok in enumerate(bio_seq):
        # End of a span 1
        if tok == default and span_type is not None:
            # Add a span.
            span = (span_type, span_start, i)
            spans.append(span)
            span_type, span_start = None, -1

        # End of a span + start of a span!
        elif tok != default:

            tok_span_class, tok_span_type = tok.split('-')
            if span_type is None:
                span_type, span_start = tok_span_type, i
            elif tok_span_type != span_type or tok_span_class == "B":
                span = (span_type, span_start, i)
                spans.append(span)
                span_type, span_start = tok_span_type, i
        else:
            pass
    # end condition
    if span_type is not None:
        span = (span_type, span_start, len(bio_seq))
        spans.append(span)

    return spans


def get_spans_from_bmes(seq: list) -> list[tuple[str, int, int]]:
    """
    :param seq: ["B", "E", "S", "B", "E","B","M","E"]
    :return:
    ([('BE', 0, 2), ('S', 2, 3), ('BE', 3, 5), ('BME', 5, 8)],
     ['BE', 'BE', 'S', 'BE', 'BE', 'BME', 'BME', 'BME'])
    """
    spans = []
    w_start = 0
    chunk = None
    tag = ""

    for i, tok in enumerate(seq):
        tag += tok
        if tok == "S":
            chunk = ("S", i, i + 1)
            spans.append(chunk)

            tag = ""
        if tok == "B":
            w_start = i
        if tok == "E":
            chunk = (tag, w_start, i + 1)
            spans.append(chunk)

            tag = ""

    return spans


class Span:
    def __init__(
        self,
        span_text: Optional[str] = None,
        span_tag: Optional[str] = None,
        span_pos: Optional[tuple] = None,
        span_capitalness: Optional[str] = None,
        span_position: Optional[float] = None,
        span_chars: Optional[int] = None,
        span_tokens: Optional[int] = None,
        span_econ: Optional[float] = None,
        span_efre: Optional[float] = None,
        sample_id: Optional[int] = None,
        span_test_freq: Optional[float] = None,
        span_train_freq: Optional[float] = None,
    ):
        self.span_text = span_text
        self.span_tag = span_tag
        self.span_pos = span_pos
        self.span_capitalness = span_capitalness
        self.span_position = span_position
        self.span_chars = span_chars
        self.span_tokens = span_tokens
        self.span_econ = span_econ
        self.span_efre = span_efre
        self.sample_id = sample_id
        self.span_test_freq = span_test_freq
        self.span_train_freq = span_train_freq


class SpanOps:
    def __init__(self, resources: dict[str, Any] = {}):
        self.resources = resources

    @abc.abstractmethod
    def get_spans(self, tags: list, seq: Optional[list] = None) -> list[Span]:
        """Returns the task type of this processor."""
        ...

    @classmethod
    def get_matched_spans(
        cls,
        spans_a: list[Span],
        spans_b: list[Span],
        activate_features: list = ["span_text"],
    ) -> tuple[list[int], list[int], list[Span]]:
        """Return matched spans based on given conditions and two span lists"""

        def is_equal(dict_a, dict_b, key):
            return True if getattr(dict_a, key) == getattr(dict_b, key) else False

        matched_a_index = []
        matched_b_index = []
        matched_spans = []

        for idx, span_dic_a in enumerate(spans_a):
            for idy, span_dic_b in enumerate(spans_b):
                if all(
                    [
                        is_equal(span_dic_a, span_dic_b, feature)
                        for feature in activate_features
                    ]
                ):
                    matched_a_index.append(idx)
                    matched_b_index.append(idy)
                    matched_spans.append(
                        span_dic_a
                    )  # return the matched a span as default this should be generalized
        return matched_a_index, matched_b_index, matched_spans


class NgramSpanOps(SpanOps):
    def __init__(self, resources: dict[str, Any] = {}, n_grams: list = [1, 2]):
        super().__init__(resources)
        self.n_grams = n_grams

    def get_spans(self, tags: list, seq: Optional[list] = None) -> list[Span]:
        span_dics = []
        deduplication: dict[tuple, int] = {}
        for k in self.n_grams:

            for i, tok in enumerate(tags):
                if i + k > len(tags):
                    break
                span = " ".join(tags[i : i + k])
                start_ind = i
                end_ind = i + k
                if (span, start_ind, end_ind) in deduplication.keys():
                    continue
                deduplication[(span, start_ind, end_ind)] = 1
                span_dic = Span(
                    span_text=span,
                    span_tag=span
                    if "span_tag" not in self.resources.keys()
                    else self.resources["span_tag"](span),
                    span_pos=(start_ind, end_ind),
                    span_capitalness=cap_feature(span),  # type: ignore
                    span_position=start_ind * 1.0 / len(tags),  # type: ignore
                    span_chars=len(span),
                    span_tokens=len(span.split(" ")),
                    span_test_freq=0
                    if "ref_test_freq" not in self.resources.keys()
                    else self.resources["ref_test_freq"].get(span, 0),  # type: ignore
                    span_train_freq=0
                    if "fre_dic" not in self.resources.keys()
                    or self.resources["fre_dic"] is None
                    else self.resources["fre_dic"].get(tok, 0),  # type: ignore
                )
                # Save the features
                span_dics.append(span_dic)

        return span_dics


class BIOSpanOps(SpanOps):
    def get_spans(self, tags: list, seq: Optional[list] = None) -> list[Span]:
        """Generate a list of spans:Span based a sequence of tokens:str"""
        default = 'O'
        if seq is None:
            seq = tags
        # idx_to_tag = {idx: tag for tag, idx in tags.items()}
        spans = []
        span_type, span_start = None, -1
        for i, tok in enumerate(tags):
            # End of a span 1
            if tok == default and span_type is not None:
                # Add a span.
                span_text = " ".join(seq[span_start:i])

                span = Span(
                    span_text=span_text,
                    span_tag=span_type,
                    span_pos=(span_start, i),
                    span_capitalness=cap_feature(span_text),
                    span_position=i * 1.0 / len(seq),
                    span_chars=len(span_text),
                    span_tokens=len(span_text.split(" ")),
                )
                if "has_stats" in self.resources.keys() and self.resources["has_stats"]:
                    lower_tag = span.span_tag.lower()
                    lower_text = span.span_text.lower()
                    span.span_econ = 0
                    if (
                        span.span_text in self.resources["econ_dic"]
                        and lower_tag in self.resources["econ_dic"][lower_text]
                    ):
                        span.span_econ = float(
                            self.resources["econ_dic"][lower_text][lower_tag]
                        )
                    span.span_efre = self.resources["efre_dic"].get(  # type: ignore
                        lower_text, 0.0
                    )

                spans.append(span)
                span_type, span_start = None, -1

            # End of a span + start of a span!
            elif tok != default:

                tok_span_class, tok_span_type = tok.split('-')
                if span_type is None:
                    span_type, span_start = tok_span_type, i
                elif tok_span_type != span_type or tok_span_class == "B":
                    span_text = " ".join(seq[span_start:i])
                    span = Span(
                        span_text=span_text,
                        span_tag=span_type,
                        span_pos=(span_start, i),
                        span_capitalness=cap_feature(span_text),
                        span_position=i * 1.0 / len(seq),
                        span_chars=len(span_text),
                        span_tokens=len(span_text.split(" ")),
                    )
                    if (
                        "has_stats" in self.resources.keys()
                        and self.resources["has_stats"]
                    ):
                        lower_tag = span.span_tag.lower()
                        lower_text = span.span_text.lower()
                        span.span_econ = 0
                        if (
                            span.span_text in self.resources["econ_dic"]
                            and lower_tag in self.resources["econ_dic"][lower_text]
                        ):
                            span.span_econ = float(
                                self.resources["econ_dic"][lower_text][lower_tag]
                            )
                        span.span_efre = self.resources["efre_dic"].get(  # type: ignore
                            lower_text, 0.0
                        )
                    spans.append(span)
                    span_type, span_start = tok_span_type, i
            else:
                pass
        # end condition
        if span_type is not None:
            span_text = " ".join(seq[span_start : len(tags)])
            span = Span(
                span_text=span_text,
                span_tag=span_type,
                span_pos=(span_start, len(tags)),
                span_capitalness=cap_feature(span_text),
                span_position=len(tags) * 1.0 / len(seq),
                span_chars=len(span_text),
                span_tokens=len(span_text.split(" ")),
            )

            if "has_stats" in self.resources.keys() and self.resources["has_stats"]:
                lower_tag = (
                    span.span_tag.lower()
                    if span.span_tag is not None
                    else span.span_tag
                )
                lower_text = (
                    span.span_text.lower()
                    if span.span_text is not None
                    else span.span_text
                )
                span.span_econ = 0
                if (
                    span.span_text in self.resources["econ_dic"]
                    and lower_tag in self.resources["econ_dic"][lower_text]
                ):
                    span.span_econ = float(
                        self.resources["econ_dic"][lower_text][lower_tag]
                    )
                span.span_efre = self.resources["efre_dic"].get(  # type: ignore
                    lower_text, 0.0
                )
            spans.append(span)

        return spans
