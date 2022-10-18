"""Tests for explainaboard.loaders.language_modeling."""

from __future__ import annotations

import unittest

from explainaboard.constants import TaskType
from explainaboard.loaders.language_modeling import LanguageModelingLoader
from explainaboard.loaders.loader_factory import get_loader_class


class LanguageModelingLoaderTest(unittest.TestCase):
    def test_get_loader_class(self) -> None:
        self.assertIs(
            get_loader_class(TaskType.language_modeling), LanguageModelingLoader
        )
