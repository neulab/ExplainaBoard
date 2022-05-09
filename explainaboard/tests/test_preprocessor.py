import unittest

from explainaboard.utils.preprocessor import ExtractiveQAPreprocessor, MapPreprocessor


class TestExampleCode(unittest.TestCase):
    """
    This tests preprocessor class
    """

    def test_mlqa_preprocessor(self):
        """
        This tests the MLQAPreprocess
        """

        en_preprocessor = ExtractiveQAPreprocessor(language='en')
        text = "This is a boring movie."
        text_processed = en_preprocessor(text)
        self.assertEqual(text_processed, "this is boring movie")

        zh_preprocessor = ExtractiveQAPreprocessor(language='zh')
        text = "这一部电影看着很无聊"
        text_processed = zh_preprocessor(text)
        self.assertEqual(text_processed, "这 一 部 电 影 看 着 很 无 聊")

    def test_map_preprocessor(self):
        dictionary = {"aaaa": "a", "bbbb": "b"}
        my_preprocessor = MapPreprocessor(resources={"dictionary": dictionary})

        text = "aaaa"
        res = my_preprocessor(text)
        self.assertEqual(res, "a")

        text = "ab"
        res = my_preprocessor(text)
        self.assertEqual(res, "ab")
