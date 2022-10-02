"""Tests for explainaboard.loaders.loader_factory."""


import unittest

from explainaboard.constants import TaskType
from explainaboard.loaders.conditional_generation import ConditionalGenerationLoader
from explainaboard.loaders.loader_factory import get_loader_class


class LoaderFactoryTest(unittest.TestCase):
    def test_get_loader_class(self) -> None:
        self.assertIs(
            get_loader_class(TaskType.conditional_generation),
            ConditionalGenerationLoader,
        )
