"""Tests for explainaboard.processors.chunking"""

from __future__ import annotations

import unittest

from explainaboard.constants import TaskType
from explainaboard.processors.chunking import ChunkingProcessor
from explainaboard.processors.processor_factory import get_processor_class


class ChunkingProcessorTest(unittest.TestCase):
    def test_get_processor_class(self) -> None:
        self.assertIs(get_processor_class(TaskType.chunking), ChunkingProcessor)
