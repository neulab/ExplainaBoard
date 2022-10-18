"""Tests for explainaboard.loaders.loader_factory."""

from __future__ import annotations

import unittest

from explainaboard.constants import TaskType
from explainaboard.loaders.conditional_generation import ConditionalGenerationLoader
from explainaboard.loaders.loader_factory import get_loader_class


# NOTE(odashi): Testing get_loader_class function should be plasced at unit test files
# for each Loader subclass.
class LoaderFactoryTest(unittest.TestCase):
    def test_get_loader_class(self) -> None:
        # This test checks if only the function is exposed appropriately.
        self.assertIs(
            get_loader_class(TaskType.conditional_generation),
            ConditionalGenerationLoader,
        )
