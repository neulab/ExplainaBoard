"""Tests for explainaboard.loaders.qa_open_domain."""

from __future__ import annotations

import unittest

from explainaboard.constants import TaskType
from explainaboard.loaders.loader_factory import get_loader_class
from explainaboard.loaders.qa_open_domain import QAOpenDomainLoader


class QAOpenDomainLoaderTest(unittest.TestCase):
    def test_get_loader_class(self) -> None:
        self.assertIs(get_loader_class(TaskType.qa_open_domain), QAOpenDomainLoader)
