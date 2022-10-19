"""Tests for explainaboard.loaders.argument_pair_extraction."""

from __future__ import annotations

import unittest

from explainaboard.constants import TaskType
from explainaboard.loaders.argument_pair_extraction import ArgumentPairExtractionLoader
from explainaboard.loaders.loader_factory import get_loader_class


class ArgumentPairExtractionLoaderTest(unittest.TestCase):
    def test_get_loader_class(self) -> None:
        self.assertIs(
            get_loader_class(TaskType.argument_pair_extraction),
            ArgumentPairExtractionLoader,
        )
