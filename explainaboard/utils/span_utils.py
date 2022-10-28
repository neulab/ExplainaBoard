"""Utilities for calculating things with respect to spans."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, cast, Optional

from explainaboard.analysis.feature_funcs import cap_feature


def _get_argument_label(token: str) -> str:
    return (
        token.split("-")[1] + "-" + token.split("-")[2]
        if len(token.split("-")) == 3
        else token.split("-")[1]
    )


def _get_argument_tokens(sentences, start, end):
    return sum([len(sentences[i].split(" ")) for i in range(start, end)])


def gen_argument_pairs(
    true_tags: list[str],
    pred_tags: list[str],
    sentences: Optional[list[str]] = None,
) -> tuple[set[str], set[str]] | tuple[list[ArgumentPair], list[ArgumentPair]]:
    """Generate argument pairs for argument pair extraction tasks.

    Args:
        true_tags: The true tags
        pred_tags: The predicted tags
        sentences: A list of sentences

    Returns:
        A tuple of gold spans and predicted spans.
    """
    # TODO(gneubig): Because this has two return types, it'd probably be better to
    #                have two different functions for the different cases.
    reply_dict: dict[int, str] = {}
    reply_pred_dict: dict[int, str] = {}
    gold_spans = []
    pred_spans = []
    gold_spans_set = set()
    pred_spans_set = set()

    for token_idx, token in enumerate(true_tags):

        gold_label = _get_argument_label(token)
        prefix = token.split("-")[0]

        next_label = (
            _get_argument_label(true_tags[token_idx + 1])
            if token_idx + 1 < len(true_tags)
            else "O"
        )

        pred_label = pred_tags[token_idx]
        next_pred_label = (
            pred_tags[token_idx + 1] if token_idx + 1 < len(pred_tags) else "O"
        )

        if prefix == "Reply":
            if gold_label.startswith("B-"):
                start = token_idx
            if (gold_label.startswith("B-") or gold_label.startswith("I-")) and (
                next_label.startswith("O") or next_label.startswith("B")
            ):
                end = token_idx
                pair_idx = int(gold_label[2:])
                if pair_idx not in reply_dict.keys():
                    reply_dict[pair_idx] = str(start) + "|" + str(end)
                else:
                    reply_dict[pair_idx] += "||" + str(start) + "|" + str(end)

            if pred_label.startswith("B-"):
                start_pred = token_idx
            if (pred_label.startswith("B-") or pred_label.startswith("I-")) and (
                next_pred_label.startswith("O") or next_pred_label.startswith("B")
            ):
                end_pred = token_idx
                pair_idx = int(pred_label[2:])
                if pair_idx not in reply_pred_dict.keys():
                    reply_pred_dict[pair_idx] = str(start_pred) + "|" + str(end_pred)
                else:
                    reply_pred_dict[pair_idx] += (
                        "||" + str(start_pred) + "|" + str(end_pred)
                    )

    for token_idx, token in enumerate(true_tags):

        gold_label = _get_argument_label(token)
        prefix = token.split("-")[0]
        next_label = (
            _get_argument_label(true_tags[token_idx + 1])
            if token_idx + 1 < len(true_tags)
            else "O"
        )

        pred_label = pred_tags[token_idx]
        next_pred_label = (
            pred_tags[token_idx + 1] if token_idx + 1 < len(pred_tags) else "O"
        )

        if prefix == "Review":
            if gold_label.startswith("B-"):
                start = token_idx
            if (gold_label.startswith("B-") or gold_label.startswith("I-")) and (
                next_label.startswith("O") or next_label.startswith("B")
            ):
                end = token_idx
                pair_idx = int(gold_label[2:])
                if pair_idx in reply_dict:
                    replies = reply_dict[pair_idx]
                    for reply in replies.split("||"):
                        reply_start_str, reply_end_str = reply.split("|")
                        reply_start = int(reply_start_str)
                        reply_end = int(reply_end_str)

                        if sentences is not None:
                            block_pos = (start, end, reply_start, reply_end)
                            block_text = (
                                "|".join(sentences[start:end])
                                + "||"
                                + "|".join(sentences[reply_start:reply_end])
                            )
                            block = ArgumentPair(
                                block_text=block_text,
                                block_tag="1",
                                block_pos=block_pos,
                                block_review_sentences=end - start + 1,
                                block_review_tokens=_get_argument_tokens(
                                    sentences, start, end
                                ),
                                block_review_position=start * 1.0 / len(sentences),
                                block_reply_sentences=reply_end - reply_start + 1,
                                block_reply_tokens=_get_argument_tokens(
                                    sentences, reply_start, reply_end
                                ),
                                block_reply_position=reply_start * 1.0 / len(sentences),
                            )
                            gold_spans.append(block)
                        gold_spans_set.add(
                            f"{start}-{end}-{reply_start}" f"-{reply_end}"
                        )

            if pred_label.startswith("B-"):
                start_pred = token_idx
            if (pred_label.startswith("B-") or pred_label.startswith("I-")) and (
                next_pred_label.startswith("O") or next_pred_label.startswith("B")
            ):
                end_pred = token_idx
                pair_idx = int(pred_label[2:])
                if pair_idx in reply_pred_dict:
                    replies_pred = reply_pred_dict[pair_idx]
                    for reply_pred in replies_pred.split("||"):
                        reply_start_pred_str, reply_end_pred_str = reply_pred.split("|")
                        reply_start_pred = int(reply_start_pred_str)
                        reply_end_pred = int(reply_end_pred_str)
                        if sentences is not None:
                            block_pos = (
                                start_pred,
                                end_pred,
                                reply_start_pred,
                                reply_end_pred,
                            )
                            block_text = (
                                "|".join(sentences[start_pred:end_pred])
                                + "||"
                                + "|".join(sentences[reply_start_pred:reply_end_pred])
                            )

                            block = ArgumentPair(
                                block_text=block_text,
                                block_tag="1",
                                block_pos=block_pos,
                                block_review_sentences=end_pred - start_pred + 1,
                                block_review_tokens=_get_argument_tokens(
                                    sentences, start_pred, end_pred
                                ),
                                block_review_position=start_pred * 1.0 / len(sentences),
                                block_reply_sentences=reply_end_pred
                                - reply_start_pred
                                + 1,
                                block_reply_tokens=_get_argument_tokens(
                                    sentences, reply_start_pred, reply_end_pred
                                ),
                                block_reply_position=reply_start_pred
                                * 1.0
                                / len(sentences),
                            )

                            pred_spans.append(block)
                        pred_spans_set.add(
                            f"{start_pred}-{end_pred}-{reply_start_pred}"
                            f"-{reply_end_pred}"
                        )

    if sentences is not None:
        return gold_spans, pred_spans
    else:
        return gold_spans_set, pred_spans_set


@dataclass
class ArgumentPair:
    """A data structure for the argument pair extraction task.

    Args:
        block_text: surface string a block of text
        block_tag: the tag of a block
        block_pos: the position of a block
        block_review_sentences: the number of review sentence
        block_review_tokens: the number of review tokens
        block_review_position: the relative position of review block
        block_reply_sentences: the number of reply sentence
        block_reply_tokens: the number of reply tokens
        block_reply_position: the relative position of reply block
        sample_id: the id of samples where a span is located
    """

    block_text: Optional[str] = None
    block_tag: Optional[str] = None
    block_pos: Optional[tuple[int, int, int, int]] = None
    block_review_sentences: Optional[float] = None
    block_review_tokens: Optional[float] = None
    block_review_position: Optional[float] = None
    block_reply_sentences: Optional[float] = None
    block_reply_tokens: Optional[float] = None
    block_reply_position: Optional[float] = None
    sample_id: Optional[int] = None


class ArgumentPairOps:
    """Operations over argument pairs."""

    def __init__(
        self, resources: dict[str, Any] | None = None, match_type: Optional[str] = None
    ) -> None:
        """Operations over argument pair.

        Args:
            resources: Resources to be used in calculation.
            match_type: Not used by this class.
        """
        self.resources = resources or {}
        self.match_type: Optional[str] = None
        self.match_func = None

    def get_argument_pairs(
        self,
        true_tags: list[str],
        pred_tags: list[str],
        sentences: list[str],
    ) -> tuple[list[ArgumentPair], list[ArgumentPair]]:
        """Generate argument pairs.

        Args:
            true_tags: The actual tags.
            pred_tags: Predicted tags.
            sentences: Sentences.

        Returns:
            A list of gold spans and predicted spans.
        """
        gold_spans, pred_spans = gen_argument_pairs(true_tags, pred_tags, sentences)
        gold_spans_list = cast(list[ArgumentPair], gold_spans)
        pred_spans_list = cast(list[ArgumentPair], pred_spans)
        return gold_spans_list, pred_spans_list


@dataclass
class Span:
    """A data structure representing a span in a sentence.

    Args:
        span_text: surface string a span
        span_tag: the tag of a span
        span_matched: whether span could be matched
        span_pos: the position of a span
        span_char_pos: the position of a span in characters in the string
        span_capitalness: span capital features
        span_rel_pos: the relative position of a span in a sequence
        span_chars: the number of characters of a span
        span_tokens: the number of tokens of a span
        span_econ: the consistency of span label in training set
        span_efre: the frequency of a span in training set
        sample_id: the id of samples where a span is located
        span_test_freq: the frequency of span in test set
        span_train_freq:  the frequency of span in training set (TODO: duplicated?)
    """

    span_text: Optional[str] = None
    span_tag: Optional[str] = None
    span_matched: int = 0
    span_pos: Optional[tuple[int, int]] = None
    span_char_pos: Optional[tuple[int, int]] = None
    span_capitalness: Optional[str] = None
    span_rel_pos: Optional[float] = None
    span_chars: Optional[int] = None
    span_tokens: Optional[int] = None
    span_econ: Optional[float] = None
    span_efre: Optional[float] = None
    sample_id: Optional[int] = None
    span_test_freq: Optional[float] = None
    span_train_freq: Optional[float] = None

    @property
    def get_span_tag(self):
        """Get the tag of the span."""
        return self.span_tag

    @property
    def get_span_text(self):
        """Get the text of the span."""
        return self.span_text


class SpanOps:
    """Operations over spans."""

    def __init__(
        self, resources: dict[str, Any] | None = None, match_type: Optional[str] = None
    ):
        """Constructor.

        Args:
            resources: Resources used in performing the operations (e.g. statistics).
            match_type: The type of matching to be performed when checking whether spans
              match for the purpose of measuring accuracy, etc.
              * "tag": match tags between spans
              * "text": match text between spans
              * "text_tag": match both the text and the tag
              * "position": match only the position
              * "position_tag": match both the position and the tag
        """
        self.resources = resources or {}
        self.match_type: str = (
            self.default_match_type() if match_type is None else match_type
        )
        self.match_func = self._get_match_funcs()[self.match_type]

    def set_resources(self, resources: dict[str, Any]) -> dict[str, Any]:
        """Set the resources used for span operations.

        Args:
            resources: A dictionary of resources to be used

        Returns:
            The resources that are set
        """
        self.resources = resources
        return resources

    @classmethod
    def default_match_type(cls) -> str:
        """Which matching type to use by default."""
        return "tag"

    def set_match_type(self, match_type) -> str:
        """Set the matching type to be used."""
        self.match_type = match_type
        self.match_func = self._get_match_funcs()[self.match_type]
        return self.match_type

    def _get_match_funcs(self):
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

        return {
            "tag": span_tag_match,
            "text": span_text_match,
            "text_tag": span_text_tag_match,
            "position": span_position_match,
            "position_tag": span_position_tag_match,
        }

    def _create_span(
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
                f"{lower_text}|||{lower_tag}", 0.0
            )
            span.span_efre = self.resources["efre_dic"].get(lower_text, 0.0)
        return span

    def get_spans(self, tags: list[str], toks: list[str]) -> list[Span]:
        """Return spans from a sequence of tags and tokens.

        Args:
            tags: A list of tags on each token.
            toks: A list of tokens.

        Returns:
            A list of spans extracted from the tags.
        """
        if len(tags) != len(toks):
            raise ValueError(f"length of tags and toks not same\n{tags}\n{toks}")
        spans = []
        char_starts = [0]
        span_start = -1
        for i, (tag, tok) in enumerate(zip(tags, toks)):
            char_starts.append(char_starts[-1] + len(tok) + 1)
            if self._span_ends(tags, i):
                spans.append(
                    self._create_span(tags, toks, char_starts, (span_start, i))
                )
                span_start = -1
            if self._span_starts(tags, i):
                span_start = i
        # end condition
        if span_start != -1:
            spans.append(
                self._create_span(tags, toks, char_starts, (span_start, len(toks)))
            )
        return spans

    def get_spans_simple(self, tags: list[str]) -> list[tuple[str, int, int]]:
        """Return spans from a sequence of tags only, no tokens needed.

        Args:
            tags: The tags from which to extract the spans.

        Returns:
            A list of tuples of "tag, start_position, end_position"
        """
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
        """Check whether a span ends at position i.

        Args:
            tags: The tags
            i: The position

        Returns:
            Whether a span ends there
        """
        ...

    @abc.abstractmethod
    def _span_starts(self, tags: list[str], i: int) -> bool:
        """Check whether a span starts at position i.

        Args:
            tags: The tags
            i: The position

        Returns:
            Whether a span starts there
        """
        ...

    @abc.abstractmethod
    def _span_type(self, tags: list[str], pos: tuple[int, int]) -> str:
        """Check the type of span that spans over positions pos.

        Args:
            tags: The tags used in the input
            pos: The span start and end

        Returns:
            The tag of the span
        """
        ...

    def get_matched_spans(
        self, spans_a: list[Span], spans_b: list[Span]
    ) -> tuple[list[int], list[int], list[Span], list[Span]]:
        """Get the spans that match between two lists of spans.

        Args:
            spans_a: One list.
            spans_b: The other list.

        Returns:
            A tuple consisting of:
            * "a" matched indices
            * "b" matched indices
            * "a" matched spans
            * "b" matched spans
        """
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


class BMESSpanOps(SpanOps):
    """SpanOps for BMES tagging schemes."""

    @classmethod
    def default_match_type(cls) -> str:
        """See SpanOps.default_match_type."""
        return "position_tag"

    def _span_ends(self, tags: list[str], i: int) -> bool:
        return i != 0 and tags[i] in {"B", "S"}

    def _span_starts(self, tags: list[str], i: int) -> bool:
        return tags[i] in {"B", "S"}

    def _span_type(self, tags: list[str], pos: tuple[int, int]) -> str:
        return "".join(tags[pos[0] : pos[1]])


class BIOSpanOps(SpanOps):
    """SpanOps for BIO tagging schemes."""

    _DEFAULT = "O"

    @classmethod
    def default_match_type(cls) -> str:
        """See SpanOps.default_match_type."""
        return "position_tag"

    def _span_ends(self, tags: list[str], i: int) -> bool:
        return i != 0 and tags[i - 1] != self._DEFAULT and not tags[i].startswith("I")

    def _span_starts(self, tags: list[str], i: int) -> bool:
        # The second condition is when an "I" tag is predicted without B
        return tags[i].startswith("B") or (
            tags[i].startswith("I") and (i == 0 or tags[i - 1] == self._DEFAULT)
        )

    def _span_type(self, tags: list[str], pos: tuple[int, int]) -> str:
        return tags[pos[0]].split("-")[1]
