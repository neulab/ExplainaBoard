"""Tests for explainaboard.utils.tokenizer."""

from __future__ import annotations

import unittest

from explainaboard.serialization.serializers import PrimitiveSerializer
from explainaboard.utils.tokenizer import (
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
    def test_empty(self) -> None:
        tokens = SingleSpaceTokenizer()("")
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens.strs, [""])
        self.assertEqual(tokens.positions, [0])

    def test_only_0x20(self) -> None:
        tokens = SingleSpaceTokenizer()("   ")
        self.assertEqual(len(tokens), 4)
        self.assertEqual(tokens.strs, ["", "", "", ""])
        self.assertEqual(tokens.positions, [0, 1, 2, 3])

    def test_isspace(self) -> None:
        tokens = SingleSpaceTokenizer()("\t\v \n\r\f")
        self.assertEqual(len(tokens), 2)
        self.assertEqual(tokens.strs, ["\t\v", "\n\r\f"])
        self.assertEqual(tokens.positions, [0, 3])

    def test_sentence(self) -> None:
        tokens = SingleSpaceTokenizer()("May the force be with you.")
        self.assertEqual(len(tokens), 6)
        self.assertEqual(tokens.strs, ["May", "the", "force", "be", "with", "you."])
        self.assertEqual(tokens.positions, [0, 4, 8, 14, 17, 22])

    def test_sentence_with_extra_whitespaces(self) -> None:
        tokens = SingleSpaceTokenizer()(" May  the force\nbe with you. ")
        self.assertEqual(len(tokens), 8)
        self.assertEqual(
            tokens.strs, ["", "May", "", "the", "force\nbe", "with", "you.", ""]
        )
        self.assertEqual(tokens.positions, [0, 1, 5, 6, 10, 19, 24, 29])

    def test_serialize(self) -> None:
        serializer = PrimitiveSerializer()
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
        serializer = PrimitiveSerializer()
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
