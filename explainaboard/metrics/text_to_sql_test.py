"""Tests for explainaboard.metrics.text_to_sql"""

from __future__ import annotations

import unittest

from explainaboard.metrics.text_to_sql import (
    SQLExactSetMatch,
    SQLExactSetMatchConfig,
    SQLExecution,
    SQLExecutionConfig,
)


class SQLExactSetMatchConfigTest(unittest.TestCase):
    def test_to_metric(self) -> None:
        self.assertIsInstance(
            SQLExactSetMatchConfig("Accuracy").to_metric(), SQLExactSetMatch
        )


class SQLExecutionConfigTest(unittest.TestCase):
    def test_to_metric(self) -> None:
        self.assertIsInstance(SQLExecutionConfig("Accuracy").to_metric(), SQLExecution)
