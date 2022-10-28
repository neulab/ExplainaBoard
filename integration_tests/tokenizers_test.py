from __future__ import annotations

import unittest

from explainaboard.utils.tokenizer import SacreBleuTokenizer, SingleSpaceTokenizer


class TokenizersTest(unittest.TestCase):
    def test_single_space_tokenizer(self):
        src = "this,  is an example "
        gold_toks = ["this,", "", "is", "an", "example", ""]
        gold_poss = [0, 6, 7, 10, 13, 21]
        tokenizer = SingleSpaceTokenizer()
        out_tokseq = tokenizer(src)
        self.assertEqual(gold_toks, out_tokseq.strs)
        self.assertEqual(gold_poss, out_tokseq.positions)

    def test_sacrebleu_intl_tokenizer(self):
        src = '"this," she   said, is an example.'
        gold_toks = [
            '"',
            "this",
            ",",
            '"',
            "she",
            "said",
            ",",
            "is",
            "an",
            "example",
            ".",
        ]
        gold_poss = [0, 1, 5, 6, 8, 14, 18, 20, 23, 26, 33]
        tokenizer = SacreBleuTokenizer()
        out_tokseq = tokenizer(src)
        self.assertEqual(gold_toks, out_tokseq.strs)
        self.assertEqual(gold_poss, out_tokseq.positions)
