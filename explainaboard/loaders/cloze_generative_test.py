"""Tests for explainaboard.loaders.cloze_generative."""

from __future__ import annotations

import unittest

from explainaboard.constants import TaskType
from explainaboard.loaders.cloze_generative import ClozeGenerativeLoader
from explainaboard.loaders.loader_factory import get_loader_class


class ClozeGenerativeLoaderTest(unittest.TestCase):
    def test_get_loader_class(self) -> None:
        self.assertIs(
            get_loader_class(TaskType.cloze_generative), ClozeGenerativeLoader
        )
