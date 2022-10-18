"""Tests for explainaboard.loaders.cloze_multiple_choice."""

from __future__ import annotations

import unittest

from explainaboard.constants import TaskType
from explainaboard.loaders.cloze_multiple_choice import ClozeMultipleChoiceLoader
from explainaboard.loaders.loader_factory import get_loader_class


class ClozeMultipleChoiceLoaderTest(unittest.TestCase):
    def test_get_loader_class(self) -> None:
        self.assertIs(
            get_loader_class(TaskType.cloze_mutiple_choice), ClozeMultipleChoiceLoader
        )
