"""Tests for explainaboard.analysis.feature_funcs."""

from __future__ import annotations

import unittest

from explainaboard.analysis.feature_funcs import get_basic_words


class FeatureFuncsTest(unittest.TestCase):
    def test_get_basic_words(self) -> None:
        # All examples should exactly match.

        # zero word
        self.assertEqual(get_basic_words(""), 0.0)
        self.assertEqual(get_basic_words(" "), 0.0)

        # one word
        self.assertEqual(get_basic_words("the"), 1.0)
        self.assertEqual(get_basic_words(" the"), 0.5)
        self.assertEqual(get_basic_words(" the "), 1 / 3)
        self.assertEqual(get_basic_words("USA"), 0.0)

        # two words
        self.assertEqual(get_basic_words("United States"), 0.0)
        self.assertEqual(get_basic_words("The USA"), 0.5)
        self.assertEqual(get_basic_words("The country"), 1.0)

        # check capitalization
        self.assertEqual(get_basic_words("The THE the tHE"), 1.0)

        # check punctuation
        self.assertEqual(get_basic_words("It is."), 0.5)
        self.assertEqual(get_basic_words("It is ."), 2 / 3)
        self.assertEqual(get_basic_words("It, is"), 0.5)
        self.assertEqual(get_basic_words("It , is"), 2 / 3)
