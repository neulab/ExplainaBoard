"""Tests for explainaboard.loaders.grammatical_error_correction."""

from __future__ import annotations

import unittest

from explainaboard.constants import TaskType
from explainaboard.loaders.grammatical_error_correction import (
    GrammaticalErrorCorrectionLoader,
)
from explainaboard.loaders.loader_factory import get_loader_class


class GrammaticalErrorCorrectionLoaderTest(unittest.TestCase):
    def test_get_loader_class(self) -> None:
        self.assertIs(
            get_loader_class(TaskType.grammatical_error_correction),
            GrammaticalErrorCorrectionLoader,
        )
