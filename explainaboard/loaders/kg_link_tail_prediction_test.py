"""Tests for explainaboard.loaders.kg_link_tail_prediction."""

from __future__ import annotations

import unittest

from explainaboard.constants import TaskType
from explainaboard.loaders.kg_link_tail_prediction import KgLinkTailPredictionLoader
from explainaboard.loaders.loader_factory import get_loader_class


class KgLinkTailPredictionLoaderTest(unittest.TestCase):
    def test_get_loader_class(self) -> None:
        self.assertIs(
            get_loader_class(TaskType.kg_link_tail_prediction),
            KgLinkTailPredictionLoader,
        )
