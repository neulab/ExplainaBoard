from __future__ import annotations

from collections.abc import Callable, Iterator

from lexicalrichness import LexicalRichness
import sacrebleu

from explainaboard.info import SysOutputInfo
from explainaboard.utils import basic_words
from explainaboard.utils.logging import progress
from explainaboard.utils.tokenizer import Tokenizer
from explainaboard.utils.typing_utils import unwrap


def _get_tokens(sys_info: SysOutputInfo, text: str | list[str], side: str) -> list[str]:
    if isinstance(text, list):
        return text
    elif side == 'source':
        return unwrap(sys_info.source_tokenizer)(text).strs
    elif side == 'target':
        return unwrap(sys_info.target_tokenizer)(text).strs
    else:
        raise ValueError(f'Bad side {side}')


def count_tokens(sys_info: SysOutputInfo, text: str, side: str = 'source') -> float:
    """
    Count the number of tokens in the text
    :param sys_info: system output information
    :param text: the text where the tokens should be counted
    :param side: whether to tokenize using the source or target side tokenizer.
      (set to 'source' by default as most tasks will have the same source and target
      tokenizer)
    :returns: the number of tokens in the text
    """
    return len(_get_tokens(sys_info, text, side))


def get_similarity_by_sacrebleu(text1, text2):
    # pip install sacrebleu
    references = [text1]
    hypothesis = text2
    score = sacrebleu.sentence_bleu(hypothesis, references).score

    return score


def get_basic_words(sentence: str):
    value_list = sentence.split(' ')
    n_words = len(value_list)
    n_basic_words = 0

    for word in value_list:

        lower = word.lower()
        if lower in basic_words.BASIC_WORDS:
            n_basic_words = n_basic_words + 1

    return n_basic_words * 1.0 / n_words


def get_lexical_richness(sentence: str):

    lex = LexicalRichness(sentence)
    results = 0

    try:
        results = lex.ttr
    except ZeroDivisionError:
        # Contains no effective words, return 0 instead
        pass
    finally:
        return results


def accumulate_vocab_from_samples(
    samples: Iterator, text_from_sample: Callable, tokenizer: Tokenizer
):
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
    side: str = 'source',
) -> float:
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
    side: str = 'source',
) -> int:
    num_oov = 0
    for w in _get_tokens(sys_info, text, side):
        if w not in vocab:
            num_oov += 1
    return num_oov


def feat_length_freq(
    sys_info: SysOutputInfo,
    text: str,
    length_freq: dict[int, float],
    side: str = 'source',
) -> float:
    length = len(_get_tokens(sys_info, text, side))
    return length_freq.get(length, 0.0)


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


def relative_position(
    sys_info: SysOutputInfo, text: str, word: str, side: str = 'source'
) -> float:
    """
    Return the relative position of a token within the string text with respect to the
    total number of tokens in the text. If the token is not found, return '-1'
    :param sys_info: system output information
    :param text: the text where the tokens should be counted
    :param word: the token to search for
    :param side: whether to tokenize using the source or target side tokenizer.
      (set to 'source' by default as most tasks will have the same source and target
      tokenizer)
    :returns: the relative position of the token
    """
    tokens = _get_tokens(sys_info, text, side)
    if word not in tokens:
        return -1
    else:
        return float(tokens.index(word)) / len(tokens)


def absolute_position(
    sys_info: SysOutputInfo, text: str, word: str, side: str = 'source'
) -> float:
    """
    Return the absolute position of a token within the string text. If the token is not
    found, return '-1'
    :param sys_info: system output information
    :param text: the text where the tokens should be counted
    :param word: the token to search for
    :param side: whether to tokenize using the source or target side tokenizer.
      (set to 'source' by default as most tasks will have the same source and target
      tokenizer)
    :returns: the absolute position of the token
    """
    tokens = _get_tokens(sys_info, text, side)
    if word not in tokens:
        return -1
    else:
        return float(tokens.index(word)) / len(tokens)
