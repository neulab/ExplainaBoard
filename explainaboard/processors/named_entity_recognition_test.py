"""Tests for explainaboard.processors.named_entity_recognition"""

from __future__ import annotations

import unittest

from explainaboard import TaskType
from explainaboard.processors.named_entity_recognition import NERProcessor
from explainaboard.processors.processor_registry import get_processor


class NERProcessorTest(unittest.TestCase):
    def test_get_processor(self) -> None:
        self.assertIsInstance(
            get_processor(TaskType.named_entity_recognition.value), NERProcessor
        )
