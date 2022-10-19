"""Tests for explainaboard.loaders.meta_evaluation_wmt_da."""

from __future__ import annotations

import unittest

from explainaboard.constants import TaskType
from explainaboard.loaders.loader_factory import get_loader_class
from explainaboard.loaders.meta_evaluation_wmt_da import MetaEvaluationWMTDALoader


class NLGMetaEvaluationLoaderTest(unittest.TestCase):
    def test_get_loader_class(self) -> None:
        self.assertIs(
            get_loader_class(TaskType.meta_evaluation_wmt_da), MetaEvaluationWMTDALoader
        )
