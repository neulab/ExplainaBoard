"""Tests for explainaboard.utils.preprocessor."""

from __future__ import annotations

import unittest

from explainaboard.utils.preprocessor import ExtractiveQAPreprocessor, MapPreprocessor


class ExtractiveQAPreprocessorTest(unittest.TestCase):
    def test_non_mixed_segmentation_languages(self):
        en_preprocessor = ExtractiveQAPreprocessor(language="en")
        text = "This is a boring movie."
        text_processed = en_preprocessor(text)
        self.assertEqual(text_processed, "this is boring movie")

    def test_mixed_segmentation_languages(self):
        zh_preprocessor = ExtractiveQAPreprocessor(language="zh")
        text = "这一部电影看着很无聊"
        text_processed = zh_preprocessor(text)
        self.assertEqual(text_processed, "这 一 部 电 影 看 着 很 无 聊")


class MapPreprocessorTest(unittest.TestCase):
    def test_call(self):
        dictionary = {"aaaa": "a", "bbbb": "b"}
        preprocessor = MapPreprocessor(dictionary=dictionary)
        self.assertEqual(preprocessor("aaaa"), "a")
        self.assertEqual(preprocessor("xbbbb"), "xbbbb")
