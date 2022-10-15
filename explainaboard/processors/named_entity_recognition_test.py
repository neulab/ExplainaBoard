"""Tests for explainaboard.processors.named_entity_recognition"""

from __future__ import annotations

import unittest

from explainaboard.constants import TaskType
from explainaboard.processors.named_entity_recognition import NERProcessor
from explainaboard.processors.processor_factory import get_processor_class


class NERProcessorTest(unittest.TestCase):
    def test_get_processor_class(self) -> None:
        self.assertIs(
            get_processor_class(TaskType.named_entity_recognition), NERProcessor
        )
