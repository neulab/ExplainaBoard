from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from lexicalrichness import LexicalRichness
import sacrebleu
from tqdm import tqdm

from explainaboard.utils import basic_words
from explainaboard.utils.tokenizer import Tokenizer


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

    # print(f"-------\n{sentence}\n")
    lex = LexicalRichness(sentence)
    results = 0

    try:
        results = lex.ttr
    except ZeroDivisionError:
        print(
            f'the sentence "{sentence}" contain no effective words, we will return 0 '
            'instead!'
        )
    finally:
        return results


def accumulate_vocab_from_samples(
    samples: Iterator, text_from_sample: Callable, tokenizer: Tokenizer
):
    vocab: dict[str, int] = {}
    for sample in tqdm(samples):
        for w in tokenizer(text_from_sample(sample)):
            vocab[w] = vocab.get(w, 0) + 1
    # the rank of each word based on its frequency
    sorted_dict = {
        key: rank
        for rank, key in enumerate(sorted(set(vocab.values()), reverse=True), 1)
    }
    vocab_rank = {k: sorted_dict[v] for k, v in vocab.items()}
    return {
        "vocab": vocab,
        "vocab_rank": vocab_rank,
    }


def feat_freq_rank(
    existing_features: dict,
    statistics: Any,
    text_from_sample: Callable,
    tokenizer: Tokenizer,
):
    fre_rank = 0

    tokens = tokenizer(text_from_sample(existing_features))
    for w in tokens:
        if w not in statistics['vocab_rank']:
            fre_rank += len(statistics['vocab_rank'])
        else:
            fre_rank += statistics['vocab_rank'][w]

    return fre_rank * 1.0 / len(tokens)


def feat_num_oov(
    existing_features: dict,
    statistics: Any,
    text_from_sample: Callable,
    tokenizer: Tokenizer,
):
    num_oov = 0
    for w in tokenizer(text_from_sample(existing_features)):
        if w not in statistics['vocab'].keys():
            num_oov += 1
    return num_oov
