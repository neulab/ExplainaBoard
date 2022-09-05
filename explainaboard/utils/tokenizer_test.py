"""Tests for explainaboard.utils.tokenizer."""

import unittest

from explainaboard.utils.tokenizer import (
    get_tokenizer_serializer,
    MLQAMixTokenizer,
    SacreBleuTokenizer,
    SingleSpaceTokenizer,
    TokenSeq,
)


class TokenSeqTest(unittest.TestCase):
    def test_data(self) -> None:
        seq = TokenSeq(["foo", "bar"], [0, 4])
        self.assertEqual(seq.strs, ["foo", "bar"])
        self.assertEqual(seq.positions, [0, 4])

    def test_sequence_interface(self) -> None:
        seq = TokenSeq(["foo", "bar"], [0, 4])
        self.assertEqual(seq[0], "foo")
        self.assertEqual(seq[:1], ["foo"])
        self.assertEqual(len(seq), 2)

        it = iter(seq)
        self.assertEqual(next(it), "foo")
        self.assertEqual(next(it), "bar")
        with self.assertRaises(StopIteration):
            next(it)

    def test_invalid_data(self) -> None:
        with self.assertRaisesRegex(ValueError, r"^strs and positions"):
            TokenSeq(["foo"], [0, 4])

    def test_from_orig_and_tokens(self) -> None:
        seq = TokenSeq.from_orig_and_tokens("foo bar", ["foo", "bar"])
        self.assertEqual(seq.strs, ["foo", "bar"])
        self.assertEqual(seq.positions, [0, 4])

    def test_from_orig_and_tokens_invalid(self) -> None:
        with self.assertRaisesRegex(ValueError, r"^Could not find"):
            TokenSeq.from_orig_and_tokens("foo bar", ["baz"])


class TokenizerSerializerTest(unittest.TestCase):
    def test_serialize(self) -> None:
        serializer = get_tokenizer_serializer()
        self.assertEqual(
            serializer.serialize(SingleSpaceTokenizer()),
            {"cls_name": "SingleSpaceTokenizer"},
        )
        self.assertEqual(
            serializer.serialize(SacreBleuTokenizer()),
            {"cls_name": "SacreBleuTokenizer", "variety": "intl"},
        )
        self.assertEqual(
            serializer.serialize(MLQAMixTokenizer()),
            {"cls_name": "MLQAMixTokenizer"},
        )

    def test_deserialize(self) -> None:
        serializer = get_tokenizer_serializer()
        self.assertIsInstance(
            serializer.deserialize({"cls_name": "SingleSpaceTokenizer"}),
            SingleSpaceTokenizer,
        )
        self.assertIsInstance(
            serializer.deserialize(
                {"cls_name": "SacreBleuTokenizer", "variety": "intl"}
            ),
            SacreBleuTokenizer,
        )
        self.assertIsInstance(
            serializer.deserialize({"cls_name": "MLQAMixTokenizer"}),
            MLQAMixTokenizer,
        )


class TokenizersTest(unittest.TestCase):
    def test_single_space_tokenizer(self):
        src = 'this,  is an example '
        gold_toks = ['this,', '', 'is', 'an', 'example', '']
        gold_poss = [0, 6, 7, 10, 13, 21]
        tokenizer = SingleSpaceTokenizer()
        out_tokseq = tokenizer(src)
        self.assertEqual(gold_toks, out_tokseq.strs)
        self.assertEqual(gold_poss, out_tokseq.positions)

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
        gold_poss = [0, 1, 5, 6, 8, 14, 18, 20, 23, 26, 33]
        tokenizer = SacreBleuTokenizer()
        out_tokseq = tokenizer(src)
        self.assertEqual(gold_toks, out_tokseq.strs)
        self.assertEqual(gold_poss, out_tokseq.positions)
