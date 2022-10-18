"""Tests for explainaboard.loaders.sequence_labeling."""

from __future__ import annotations

import unittest

from explainaboard.constants import TaskType
from explainaboard.loaders.loader_factory import get_loader_class
from explainaboard.loaders.sequence_labeling import SeqLabLoader


class SeqLabLoaderTest(unittest.TestCase):
    def test_get_loader_class(self) -> None:
        self.assertIs(get_loader_class(TaskType.chunking), SeqLabLoader)
        self.assertIs(get_loader_class(TaskType.named_entity_recognition), SeqLabLoader)
        self.assertIs(get_loader_class(TaskType.word_segmentation), SeqLabLoader)
