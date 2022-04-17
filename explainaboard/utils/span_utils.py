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
        # surface string a span
        span_text: Optional[str] = None,
        # the tag of a span
        span_tag: Optional[str] = None,
        # the position of a span
        span_pos: Optional[tuple] = None,
        # span capital features
        span_capitalness: Optional[str] = None,
        # the relative position of a span in a sequence
        span_rel_pos: Optional[float] = None,
        # the number of characters of a span
        span_chars: Optional[int] = None,
        # the number of tokens of a span
        span_tokens: Optional[int] = None,
        # the consistency of span label in training set
        span_econ: Optional[float] = None,
        # the frequency of a span in training set
        span_efre: Optional[float] = None,
        # the id of samples where a span is located
        sample_id: Optional[int] = None,
        # the frequency of span in test set
        span_test_freq: Optional[float] = None,
        # the frequency of span in training set (TODO: duplicated?)
        span_train_freq: Optional[float] = None,
    ):
        self.span_text = span_text
        self.span_tag = span_tag
        self.span_pos = span_pos
        self.span_capitalness = span_capitalness
        self.span_rel_pos = span_rel_pos
        self.span_chars = span_chars
        self.span_tokens = span_tokens
        self.span_econ = span_econ
        self.span_efre = span_efre
        self.sample_id = sample_id
        self.span_test_freq = span_test_freq
        self.span_train_freq = span_train_freq

    @property
    def get_span_tag(self):
        return self.span_tag

    @property
    def get_span_text(self):
        return self.span_text


class SpanOps:
    def __init__(self, resources: dict[str, Any] = {}):
        self.resources = resources

    @abc.abstractmethod
    def get_spans(self, tags: list, seq: Optional[list] = None) -> list[Span]:
        """Return spans from a sequence of tags and tokens"""
        ...

    @classmethod
    def get_matched_spans(
        cls,
        spans_a: list[Span],
        spans_b: list[Span],
        activate_features: list = ["span_text"],
    ) -> tuple[list[int], list[int], list[Span], list[Span]]:

        # TODO(Pengfei): add more matched condition
        def is_equal(dict_a, dict_b, key):
            return True if getattr(dict_a, key) == getattr(dict_b, key) else False

        matched_a_index = []
        matched_b_index = []
        matched_spans_a = []
        matched_spans_b = []

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
                    matched_spans_a.append(span_dic_a)
                    matched_spans_b.append(span_dic_b)
        return matched_a_index, matched_b_index, matched_spans_a, matched_spans_b


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
                    span_rel_pos=start_ind * 1.0 / len(tags),  # type: ignore
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


class BMESSpanOps(SpanOps):
    def get_spans(self, tags: list, seq: Optional[list] = None) -> list[Span]:
        """
        :param seq: ["B", "E", "S", "B", "E","B","M","E"]
        :return:
        ([('BE', 0, 2), ('S', 2, 3), ('BE', 3, 5), ('BME', 5, 8)],
         ['BE', 'BE', 'S', 'BE', 'BE', 'BME', 'BME', 'BME'])
        """
        if seq is None:
            seq = tags
        spans = []
        w_start = 0
        tag = ""

        for i, tok in enumerate(tags):
            tag += tok
            if tok == "S":

                span_text = " ".join(seq[i : i + 1])
                span = Span(
                    span_text=span_text,
                    span_tag="S",
                    span_pos=(i, i + 1),
                    span_rel_pos=i * 1.0 / len(tags),
                    span_chars=len(span_text),
                )
                if "has_stats" in self.resources.keys() and self.resources["has_stats"]:
                    lower_tag = span.span_tag
                    lower_text = span.span_text
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
                tag = ""
            if tok == "B":
                w_start = i
            if tok == "E":
                span_text = " ".join(seq[w_start : i + 1])
                span = Span(
                    span_text=span_text,
                    span_tag=tag,
                    span_pos=(w_start, i + 1),
                    span_rel_pos=w_start * 1.0 / len(tags),
                    span_chars=len(span_text),
                )
                if "has_stats" in self.resources.keys() and self.resources["has_stats"]:
                    lower_tag = span.span_tag
                    lower_text = span.span_text
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
                tag = ""

        return spans


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
                    span_rel_pos=i * 1.0 / len(seq),
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
                        span_rel_pos=i * 1.0 / len(seq),
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
                span_rel_pos=len(tags) * 1.0 / len(seq),
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
