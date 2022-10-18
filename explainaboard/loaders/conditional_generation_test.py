"""Tests for explainaboard.loaders.conditional_generation."""

from __future__ import annotations

import unittest

from explainaboard.constants import TaskType
from explainaboard.loaders.conditional_generation import (
    ConditionalGenerationLoader,
    MachineTranslationLoader,
    SummarizationLoader,
)
from explainaboard.loaders.loader_factory import get_loader_class


class ConditionalGenerationLoaderTest(unittest.TestCase):
    def test_get_loader_class(self) -> None:
        self.assertIs(
            get_loader_class(TaskType.conditional_generation),
            ConditionalGenerationLoader,
        )


class SummarizationLoaderTest(unittest.TestCase):
    def test_get_loader_class(self) -> None:
        self.assertIs(get_loader_class(TaskType.summarization), SummarizationLoader)


class MachineTranslationLoaderTest(unittest.TestCase):
    def test_get_loader_class(self) -> None:
        self.assertIs(
            get_loader_class(TaskType.machine_translation), MachineTranslationLoader
        )
