from __future__ import annotations

import abc
from dataclasses import dataclass
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


@dataclass
class Span:

    # surface string a span
    span_text: Optional[str] = None
    # the tag of a span
    span_tag: Optional[str] = None
    # whether span could be matched
    span_matched: int = 0
    # the position of a span
    span_pos: Optional[tuple[int, int]] = None
    # the position of a span in characters in the string
    span_char_pos: Optional[tuple[int, int]] = None
    # span capital features
    span_capitalness: Optional[str] = None
    # the relative position of a span in a sequence
    span_rel_pos: Optional[float] = None
    # the number of characters of a span
    span_chars: Optional[int] = None
    # the number of tokens of a span
    span_tokens: Optional[int] = None
    # the consistency of span label in training set
    span_econ: Optional[float] = None
    # the frequency of a span in training set
    span_efre: Optional[float] = None
    # the id of samples where a span is located
    sample_id: Optional[int] = None
    # the frequency of span in test set
    span_test_freq: Optional[float] = None
    # the frequency of span in training set (TODO: duplicated?)
    span_train_freq: Optional[float] = None

    @property
    def get_span_tag(self):
        return self.span_tag

    @property
    def get_span_text(self):
        return self.span_text


class SpanOps:
    def __init__(
        self, resources: dict[str, Any] | None = None, match_type: Optional[str] = None
    ):
        self.resources = resources or {}
        self.match_type: str = (
            self.default_match_type() if match_type is None else match_type
        )
        self.match_func = self.get_match_funcs()[self.match_type]

    def set_resources(self, resources: dict[str, Any]):
        self.resources = resources
        return resources

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

    def create_span(
        self,
        tags: list[str],
        toks: list[str],
        char_starts: list[int],
        pos: tuple[int, int],
    ) -> Span:
        # Add a span.
        span_text = " ".join(toks[pos[0] : pos[1]])
        span = Span(
            span_text=span_text,
            span_tag=self._span_type(tags, pos),
            span_pos=pos,
            span_char_pos=(char_starts[pos[0]], char_starts[pos[1]] - 1),
            span_capitalness=cap_feature(span_text),
            span_rel_pos=pos[0] * 1.0 / len(toks),
            span_chars=len(span_text),
            span_tokens=len(span_text.split(" ")),
        )
        if "has_stats" in self.resources.keys() and self.resources["has_stats"]:
            lower_tag = span.span_tag
            lower_text = span_text.lower()
            span.span_econ = self.resources["econ_dic"].get(
                f'{lower_text}|||{lower_tag}', 0.0
            )
            span.span_efre = self.resources["efre_dic"].get(lower_text, 0.0)
        return span

    def get_spans(self, tags: list[str], toks: list[str]) -> list[Span]:
        """Return spans from a sequence of tags and tokens"""
        if len(tags) != len(toks):
            raise ValueError(f'length of tags and toks not same\n{tags}\n{toks}')
        spans = []
        char_starts = [0]
        span_start = -1
        for i, (tag, tok) in enumerate(zip(tags, toks)):
            char_starts.append(char_starts[-1] + len(tok) + 1)
            if self._span_ends(tags, i):
                spans.append(self.create_span(tags, toks, char_starts, (span_start, i)))
                span_start = -1
            if self._span_starts(tags, i):
                span_start = i
        # end condition
        if span_start != -1:
            spans.append(
                self.create_span(tags, toks, char_starts, (span_start, len(toks)))
            )
        return spans

    def get_spans_simple(self, tags: list[str]) -> list[tuple[str, int, int]]:
        """Return spans from a sequence of tags and tokens"""
        spans = []
        span_start = -1
        for i, tag in enumerate(tags):
            if self._span_ends(tags, i):
                pos = (span_start, i)
                spans.append((self._span_type(tags, pos), pos[0], pos[1]))
                span_start = -1
            if self._span_starts(tags, i):
                span_start = i
        # end condition
        if span_start != -1:
            pos = (span_start, len(tags))
            spans.append((self._span_type(tags, pos), pos[0], pos[1]))
        return spans

    @abc.abstractmethod
    def _span_ends(self, tags: list[str], i: int) -> bool:
        ...

    @abc.abstractmethod
    def _span_starts(self, tags: list[str], i: int) -> bool:
        ...

    @abc.abstractmethod
    def _span_type(self, tags: list[str], pos: tuple[int, int]) -> str:
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


# TODO(gneubig): this is not used anywhere, so decide to keep or delete
# class NgramSpanOps(SpanOps):
#     def __init__(
#         self,
#         resources: dict[str, Any] | None = None,
#         match_type: str = "tag",
#         n_grams: list | None = None,
#     ):
#         super().__init__(resources, match_type)
#         self.n_grams = n_grams or list([1, 2])
#
#     @classmethod
#     def default_match_type(cls) -> str:
#         return "tag"
#
#     def get_spans_and_match(self, tags: list, tags_other: list):
#         def get_ngrams(tags, n_grams: list[int]):
#             spans = []
#             for k in n_grams:
#                 for i, tok in enumerate(tags):
#                     if i + k > len(tags):
#                         break
#                     span = " ".join(tags[i : i + k])
#                     spans.append((span, i, i + k))
#             return spans
#
#         def get_span_from_ngrams(ngrams, tags_other_table, tags_length):
#             span_dics = []
#             for ngram in ngrams:
#                 span = ngram[0]
#
#                 # match
#                 my_other = tags_other_table.get(span, list())
#                 matched = my_other.pop(0) if len(my_other) > 0 else -1
#
#                 span_dic = Span(
#                     span_text=span,
#                     span_tag=span
#                     if "span_tag" not in self.resources.keys()
#                     else self.resources["span_tag"](span),
#                     span_pos=(ngram[1], ngram[2]),
#                     span_matched=matched,
#                     span_capitalness=cap_feature(span),  # type: ignore
#                     span_rel_pos=ngram[2] * 1.0 / tags_length,  # type: ignore
#                     span_chars=len(span),
#                     span_tokens=len(span.split(" ")),
#                     span_test_freq=0
#                     if "ref_test_freq" not in self.resources.keys()
#                     else self.resources["ref_test_freq"].get(span, 0),  # type: ignore
#                     span_train_freq=0
#                     if "fre_dic" not in self.resources.keys()
#                     or self.resources["fre_dic"] is None
#                     else self.resources["fre_dic"].get(span, 0),  # type: ignore
#                 )
#                 # Save the features
#                 span_dics.append(span_dic)
#             return span_dics
#
#         tags_ngrams = get_ngrams(tags, self.n_grams)
#         tags_other_ngrams = get_ngrams(tags_other, self.n_grams)
#
#         # Find tokens in other set
#         tags_other_table = defaultdict(list)
#         for i, tok in enumerate(tags_other_ngrams):
#             tags_other_table[tok[0]].append(i)
#
#         # Find tokens in other set
#         tags_table = defaultdict(list)
#         for i, tok in enumerate(tags_ngrams):
#             tags_table[tok[0]].append(i)
#
#         span_dics = get_span_from_ngrams(tags_ngrams, tags_other_table, len(tags))
#         span_dics_other = get_span_from_ngrams(
#             tags_other_ngrams, tags_table, len(tags_other)
#         )
#
#         return span_dics, span_dics_other


class BMESSpanOps(SpanOps):
    @classmethod
    def default_match_type(cls) -> str:
        return "position_tag"

    def _span_ends(self, tags: list[str], i: int) -> bool:
        return i != 0 and tags[i] in {'B', 'S'}

    def _span_starts(self, tags: list[str], i: int) -> bool:
        return tags[i] in {'B', 'S'}

    def _span_type(self, tags: list[str], pos: tuple[int, int]) -> str:
        return ''.join(tags[pos[0] : pos[1]])


class BIOSpanOps(SpanOps):

    _DEFAULT = 'O'

    @classmethod
    def default_match_type(cls) -> str:
        return "position_tag"

    def _span_ends(self, tags: list[str], i: int) -> bool:
        return i != 0 and tags[i - 1] != self._DEFAULT and not tags[i].startswith('I')

    def _span_starts(self, tags: list[str], i: int) -> bool:
        # The second condition is when an "I" tag is predicted without B
        return tags[i].startswith('B') or (
            tags[i].startswith('I') and (i == 0 or tags[i - 1] == self._DEFAULT)
        )

    def _span_type(self, tags: list[str], pos: tuple[int, int]) -> str:
        return tags[pos[0]].split('-')[1]
