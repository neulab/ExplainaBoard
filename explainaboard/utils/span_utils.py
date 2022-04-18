from __future__ import annotations

import abc
from collections import defaultdict
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
        # whether span could be matched
        span_matched: int = 0,
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
        self.span_matched = span_matched
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
    def __init__(
        self, resources: dict[str, Any] = {}, match_type: Optional[str] = None
    ):
        self.resources = resources
        self.match_type: str = (
            self.default_match_type() if match_type is None else match_type
        )
        self.match_func = self.get_match_funcs()[self.match_type]

    @classmethod
    def default_match_type(cls) -> str:
        return "tag"

    def set_match_type(self, match_type) -> str:
        self.match_type = match_type
        self.match_func = self.get_match_funcs()[self.match_type]
        return self.match_type

    def get_match_funcs(self):

        match_funcs = {}

        def span_tag_match(span_a: Span, span_b: Span):
            return span_a.span_tag == span_b.span_tag

        def span_text_match(span_a: Span, span_b: Span):
            return span_a.span_text == span_b.span_text

        def span_text_tag_match(span_a: Span, span_b: Span):
            return (
                span_a.span_tag == span_b.span_tag
                and span_a.span_text == span_b.span_text
            )

        def span_position_match(span_a: Span, span_b: Span):
            return span_a.span_pos == span_b.span_pos

        def span_position_tag_match(span_a: Span, span_b: Span):
            return (
                span_a.span_tag == span_b.span_tag
                and span_a.span_pos == span_b.span_pos
            )

        match_funcs["tag"] = span_tag_match
        match_funcs["text"] = span_text_match
        match_funcs["text_tag"] = span_text_tag_match
        match_funcs["position"] = span_position_match
        match_funcs["position_tag"] = span_position_tag_match

        return match_funcs

    @abc.abstractmethod
    def get_spans(self, tags: list, seq: Optional[list] = None) -> list[Span]:
        """Return spans from a sequence of tags and tokens"""
        ...

    def get_matched_spans(
        self, spans_a: list[Span], spans_b: list[Span]
    ) -> tuple[list[int], list[int], list[Span], list[Span]]:

        # # TODO(Pengfei): add more matched condition
        # def is_equal(dict_a, dict_b, key):
        #     return True if getattr(dict_a, key) == getattr(dict_b, key) else False

        matched_a_index = []
        matched_b_index = []
        matched_spans_a = []
        matched_spans_b = []

        # return matched_a_index, matched_b_index, matched_spans_a, matched_spans_b
        # TODO(Pengfei): this part is not very efficient
        for idx, span_dic_a in enumerate(spans_a):
            for idy, span_dic_b in enumerate(spans_b):
                if span_dic_a.sample_id != span_dic_b.sample_id:
                    continue
                if self.match_func(span_dic_a, span_dic_b):
                    matched_a_index.append(idx)
                    matched_b_index.append(idy)
                    matched_spans_a.append(span_dic_a)
                    matched_spans_b.append(span_dic_b)
        return matched_a_index, matched_b_index, matched_spans_a, matched_spans_b


class NgramSpanOps(SpanOps):
    def __init__(
        self,
        resources: dict[str, Any] = {},
        match_type: str = "tag",
        n_grams: list = [1, 2],
    ):
        super().__init__(resources, match_type)
        self.n_grams = n_grams

    @classmethod
    def default_match_type(cls) -> str:
        return "tag"

    def get_spans_and_match(self, tags: list, tags_other: list):
        def get_ngrams(tags, n_grams: list[int]):
            spans = []
            for k in n_grams:
                for i, tok in enumerate(tags):
                    if i + k > len(tags):
                        break
                    span = " ".join(tags[i : i + k])
                    spans.append((span, i, i + k))
            return spans

        def get_span_from_ngrams(ngrams, tags_other_table, tags_length):
            span_dics = []
            for ngram in ngrams:
                span = ngram[0]

                # match
                my_other = tags_other_table.get(span, list())
                matched = my_other.pop(0) if len(my_other) > 0 else -1

                span_dic = Span(
                    span_text=span,
                    span_tag=span
                    if "span_tag" not in self.resources.keys()
                    else self.resources["span_tag"](span),
                    span_pos=(ngram[1], ngram[2]),
                    span_matched=matched,
                    span_capitalness=cap_feature(span),  # type: ignore
                    span_rel_pos=ngram[2] * 1.0 / tags_length,  # type: ignore
                    span_chars=len(span),
                    span_tokens=len(span.split(" ")),
                    span_test_freq=0
                    if "ref_test_freq" not in self.resources.keys()
                    else self.resources["ref_test_freq"].get(span, 0),  # type: ignore
                    span_train_freq=0
                    if "fre_dic" not in self.resources.keys()
                    or self.resources["fre_dic"] is None
                    else self.resources["fre_dic"].get(span, 0),  # type: ignore
                )
                # Save the features
                span_dics.append(span_dic)
            return span_dics

        tags_ngrams = get_ngrams(tags, self.n_grams)
        tags_other_ngrams = get_ngrams(tags_other, self.n_grams)

        # Find tokens in other set
        tags_other_table = defaultdict(list)
        for i, tok in enumerate(tags_other_ngrams):
            tags_other_table[tok[0]].append(i)

        # Find tokens in other set
        tags_table = defaultdict(list)
        for i, tok in enumerate(tags_ngrams):
            tags_table[tok[0]].append(i)

        span_dics = get_span_from_ngrams(tags_ngrams, tags_other_table, len(tags))
        span_dics_other = get_span_from_ngrams(
            tags_other_ngrams, tags_table, len(tags_other)
        )

        return span_dics, span_dics_other


class BMESSpanOps(SpanOps):
    @classmethod
    def default_match_type(cls) -> str:
        return "position_tag"

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
                    span.span_econ = self.resources["econ_dic"].get(
                        f'{lower_text}|||{lower_tag}', 0.0
                    )
                    span.span_efre = self.resources["efre_dic"].get(lower_text, 0.0)

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
                    span.span_econ = self.resources["econ_dic"].get(
                        f'{lower_text}|||{lower_tag}', 0.0
                    )
                    span.span_efre = self.resources["efre_dic"].get(lower_text, 0.0)

                spans.append(span)
                tag = ""

        return spans


class BIOSpanOps(SpanOps):
    @classmethod
    def default_match_type(cls) -> str:
        return "position_tag"

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
                    lower_tag = span.span_tag
                    lower_text = span.span_text.lower()
                    span.span_econ = self.resources["econ_dic"].get(
                        f'{lower_text}|||{lower_tag}', 0.0
                    )
                    span.span_efre = self.resources["efre_dic"].get(lower_text, 0.0)

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
                        span.span_econ = self.resources["econ_dic"].get(
                            f'{lower_text}|||{lower_tag}', 0.0
                        )
                        span.span_efre = self.resources["efre_dic"].get(lower_text, 0.0)
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
                span.span_econ = self.resources["econ_dic"].get(
                    f'{lower_text}|||{lower_tag}', 0.0
                )
                span.span_efre = self.resources["efre_dic"].get(lower_text, 0.0)
            spans.append(span)

        return spans
