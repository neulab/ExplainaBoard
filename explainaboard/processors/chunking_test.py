"""Tests for explainaboard.processors.chunking"""

from __future__ import annotations

import unittest

from explainaboard import TaskType
from explainaboard.processors.chunking import ChunkingProcessor
from explainaboard.processors.processor_registry import get_processor


class ChunkingProcessorTest(unittest.TestCase):
    def test_get_processor(self) -> None:
        self.assertIsInstance(
            get_processor(TaskType.chunking.value),
            ChunkingProcessor,
        )
