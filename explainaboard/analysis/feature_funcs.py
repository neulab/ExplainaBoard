"""Implementations of various common feature functions implemented across tasks."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from lexicalrichness import LexicalRichness
import sacrebleu

from explainaboard.info import SysOutputInfo
from explainaboard.utils import basic_words
from explainaboard.utils.logging import progress
from explainaboard.utils.tokenizer import SingleSpaceTokenizer, Tokenizer
from explainaboard.utils.typing_utils import unwrap


def _get_tokens(sys_info: SysOutputInfo, text: str | list[str], side: str) -> list[str]:
    if isinstance(text, list):
        return text
    elif side == "source":
        return unwrap(sys_info.source_tokenizer)(text).strs
    elif side == "target":
        return unwrap(sys_info.target_tokenizer)(text).strs
    else:
        raise ValueError(f"Bad side {side}")


def count_tokens(sys_info: SysOutputInfo, text: str, side: str = "source") -> float:
    """Count the number of tokens in the text.

    Args:
        sys_info: system output information
        text: the text where the tokens should be counted
        side: whether to tokenize using the source or target side tokenizer.
          (set to 'source' by default as most tasks will have the same source and target
          tokenizer)

    Returns:
        the number of tokens in the text
    """
    return len(_get_tokens(sys_info, text, side))


def get_similarity_by_sacrebleu(text1: str, text2: str) -> float:
    """Return the similarity between two texts according to sentence BLEU.

    Args:
        text1: The text to be used as the reference.
        text2: The text to be used as the hypothesis.

    Returns:
        Sentence BLEU with text1 as the reference and text2 as the hypothesis.
    """
    references = [text1]
    hypothesis = text2
    score = sacrebleu.sentence_bleu(hypothesis, references).score

    return score


def get_basic_words(text: str) -> float:
    """The ratio of "basic words" in a text according to an English basic words list.

    Args:
        text: The text from which the basic words will be calculated.

    Returns:
        The ratio of basic words.
    """
    tokens = SingleSpaceTokenizer()(text)
    assert len(tokens) > 0, f"BUG: no tokens obtained from the text: '{text}'"
    n_basic_words = sum(1 for t in tokens if t.lower() in basic_words.BASIC_WORDS)
    return n_basic_words / len(tokens)


def get_lexical_richness(text: str) -> float:
    """Return a lexical richness value according to the lexical richness library.

    Args:
        text: The text to assess

    Returns:
        The lexical richness value, or 0.0 if there are no effective words.
    """
    lex = LexicalRichness(text)
    results = 0.0

    try:
        results = lex.ttr
    except ZeroDivisionError:
        # Contains no effective words, return 0 instead
        pass
    finally:
        return results


def accumulate_vocab_from_samples(
    samples: Iterable[Any], text_from_sample: Callable[..., str], tokenizer: Tokenizer
):
    """From many samples, find the vocabulary+counts and frequency rank.

    Args:
        samples: An iterable of sample that are used in calculating a vocabulary.
        text_from_sample: A function that takes in each sample and outputs the text.
        tokenizer: The tokenizer to be applied to get tokens from the text.

    Returns:
        Two dictionaries:
            A dictionary of vocabulary item -> frequency
            A dictionary of vocabulary item -> frequency rank
    """
    vocab: dict[str, int] = {}
    for sample in progress(samples):
        for w in tokenizer(text_from_sample(sample)):
            vocab[w] = vocab.get(w, 0) + 1
    # the rank of each word based on its frequency
    sorted_dict = {
        key: rank
        for rank, key in enumerate(sorted(set(vocab.values()), reverse=True), 1)
    }
    vocab_rank = {k: sorted_dict[v] for k, v in vocab.items()}
    return vocab, vocab_rank


def feat_freq_rank(
    sys_info: SysOutputInfo,
    text: str | list[str],
    vocab_rank: dict[str, int],
    side: str = "source",
) -> float:
    """Return the average frequency rank of the tokens in the text.

    Args:
        sys_info: The system info (for tokenization)
        text: The text to assess
        vocab: The vocabulary mapping from strings to frequency rank in the training
          corpus
        side: Whether the text is from the source or target side (for tokenization)

    Returns:
        The average frequency rank of the words in text
    """
    fre_rank = 0

    tokens = _get_tokens(sys_info, text, side)
    max_rank = len(vocab_rank)
    for w in tokens:
        fre_rank += vocab_rank.get(w, max_rank)

    return fre_rank * 1.0 / len(tokens)


def feat_num_oov(
    sys_info: SysOutputInfo,
    text: str | list[str],
    vocab: dict[str, int],
    side: str = "source",
) -> int:
    """Return the number of out-of-vocabulary words in a text.

    Args:
        sys_info: The system info (for tokenization)
        text: The text to assess
        vocab: The vocabulary mapping from strings to counts in the tarining corpus
        side: Whether the text is from the source or target side (for tokenization)

    Returns:
        The number of OOVs in the text
    """
    num_oov = 0
    for w in _get_tokens(sys_info, text, side):
        if w not in vocab:
            num_oov += 1
    return num_oov


def feat_length_freq(
    sys_info: SysOutputInfo,
    text: str,
    length_freq: dict[int, float],
    side: str = "source",
) -> float:
    """A feature regarding how frequent the length is in the training corpus.

    Args:
        sys_info: Information about they system (for tokenizers).
        text: The text to measure the length of.
        length_freq: A dictionary of length -> frequency mappings.
        side: Whether this is the "source" or "target" (for tokenization)

    Returns:
        The frequency of the length.
    """
    length = len(_get_tokens(sys_info, text, side))
    return length_freq.get(length, 0.0)


def cap_feature(text: str) -> str:
    """Return a feature regarding capitalization.

    Args:
        text: a string to check for capitalization

    Returns:
        "low_caps" or "full_caps" if it's all lower- or upper-case respectively,
        "first_caps" if the first letter is caps, and "not_first_caps" otherwise.
    """
    if text.lower() == text:
        return "low_caps"
    elif text.upper() == text:
        return "full_caps"
    elif text[0].upper() == text[0]:
        return "first_caps"
    else:
        return "not_first_caps"


def relative_position(
    sys_info: SysOutputInfo, text: str, word: str, side: str = "source"
) -> float:
    """Return the relative position of a token within the string text.

    This is calculated with respect to the
    total number of tokens in the text. If the token is not found, return '-1'

    Args:
        sys_info: system output information
        text: the text where the tokens should be counted
        word: the token to search for
        side: whether to tokenize using the source or target side tokenizer.
          (set to 'source' by default as most tasks will have the same source and target
          tokenizer)

    Returns:
        the relative position of the token
    """
    tokens = _get_tokens(sys_info, text, side)
    if word not in tokens:
        return -1
    else:
        return float(tokens.index(word)) / len(tokens)


def absolute_position(
    sys_info: SysOutputInfo, text: str, word: str, side: str = "source"
) -> float:
    """Return the absolute position of a token within the string text.

    If the token is not found, return '-1'

    Args:
        sys_info: system output information
        text: the text where the tokens should be counted
        word: the token to search for
        side: whether to tokenize using the source or target side tokenizer.
          (set to 'source' by default as most tasks will have the same source and target
          tokenizer)

    Returns:
        the absolute position of the token
    """
    tokens = _get_tokens(sys_info, text, side)
    if word not in tokens:
        return -1
    else:
        return float(tokens.index(word)) / len(tokens)
