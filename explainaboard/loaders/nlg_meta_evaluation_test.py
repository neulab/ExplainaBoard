"""Tests for explainaboard.loaders.nlg_meta_evaluation."""


import unittest

from explainaboard.constants import TaskType
from explainaboard.loaders.loader_factory import get_loader_class
from explainaboard.loaders.nlg_meta_evaluation import NLGMetaEvaluationLoader


class NLGMetaEvaluationLoaderTest(unittest.TestCase):
    def test_get_loader_class(self) -> None:
        self.assertIs(
            get_loader_class(TaskType.nlg_meta_evaluation), NLGMetaEvaluationLoader
        )
