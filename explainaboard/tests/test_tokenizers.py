import unittest

from explainaboard.utils.tokenizer import SacreBleuTokenizer, SingleSpaceTokenizer


class TestTokenizers(unittest.TestCase):
    def test_single_space_tokenizer(self):
        src = 'this,  is an example '
        gold_toks = ['this,', '', 'is', 'an', 'example', '']
        tokenizer = SingleSpaceTokenizer()
        out_toks = tokenizer(src)
        self.assertEqual(gold_toks, out_toks)

    def test_sacrebleu_intl_tokenizer(self):
        src = '"this," she   said, is an example.'
        gold_toks = [
            '"',
            'this',
            ',',
            '"',
            'she',
            'said',
            ',',
            'is',
            'an',
            'example',
            '.',
        ]
        tokenizer = SacreBleuTokenizer()
        out_toks = tokenizer(src)
        self.assertEqual(gold_toks, out_toks)
