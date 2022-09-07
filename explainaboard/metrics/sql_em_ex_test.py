"""Tests for explainaboard.metrics.sql_em_ex"""

from __future__ import annotations

import unittest

from explainaboard.metrics.sql_em_ex import (
    SQLEm,
    SQLEx,
    SQLEmConfig,
    SQLEmConfig,
)


class SQLEmConfigTest(unittest.TestCase):
    def test_to_metric(self) -> None:
        self.assertIsInstance(SQLEmConfig("Accuracy").to_metric(), SQLEm)

class SQLExConfigTest(unittest.TestCase):
    def test_to_metric(self) -> None:
        self.assertIsInstance(SQLExConfig("Accuracy").to_metric(), SQLEx)
