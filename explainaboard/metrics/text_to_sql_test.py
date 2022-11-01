"""Tests for explainaboard.metrics.text_to_sql

We test the two metrics with examples in
intergration_tests.metric_test.test_sql_exactsetmatch
and intergration_tests.metric_test.test_sql_execution.
The test requires network connection.
"""

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
        self.assertIsInstance(SQLExactSetMatchConfig().to_metric(), SQLExactSetMatch)


class SQLExecutionConfigTest(unittest.TestCase):
    def test_to_metric(self) -> None:
        self.assertIsInstance(SQLExecutionConfig().to_metric(), SQLExecution)
