import unittest

from explainaboard.utils.preprocessor import MLQAPreprocessor


class TestExampleCode(unittest.TestCase):
    """
    This tests preprocessor class
    """

    def test_mlqa_preprocessor(self):
        """
        This tests the MLQAPreprocess
        """

        mlqa_preprocessor = MLQAPreprocessor()
        text = "This is a boring movie."
        text_processed = mlqa_preprocessor(text, "en")
        self.assertEqual(text_processed, "this is boring movie")

        text = "这一部电影看着很无聊"
        text_processed = mlqa_preprocessor(text, "zh")
        self.assertEqual(text_processed, "这 一 部 电 影 看 着 很 无 聊")
