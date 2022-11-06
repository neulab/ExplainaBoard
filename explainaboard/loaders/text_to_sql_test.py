"""Tests for explainaboard.loaders.text_to_sql."""

from __future__ import annotations

import unittest

from explainaboard.constants import TaskType
from explainaboard.loaders.loader_factory import get_loader_class
from explainaboard.loaders.text_to_sql import TextToSQLLoader


class TextToSQLLoaderTest(unittest.TestCase):
    def test_get_loader_class(self) -> None:
        self.assertIs(
            get_loader_class(TaskType.text_to_sql),
            TextToSQLLoader,
        )
