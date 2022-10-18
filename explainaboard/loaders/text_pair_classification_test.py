"""Tests for explainaboard.loaders.text_pair_classification."""

from __future__ import annotations

import unittest

from explainaboard.constants import TaskType
from explainaboard.loaders.loader_factory import get_loader_class
from explainaboard.loaders.text_pair_classification import TextPairClassificationLoader


class TextPairClassificationLoaderTest(unittest.TestCase):
    def test_get_loader_class(self) -> None:
        self.assertIs(
            get_loader_class(TaskType.text_pair_classification),
            TextPairClassificationLoader,
        )
