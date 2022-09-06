"""Tests for explainaboard.metrics.sql_em_ex"""

from __future__ import annotations

import unittest

from explainaboard.metrics.sql_em_ex import (
    SQLEmEx,
    SQLEmExConfig,
)


class SQLEmExConfigTest(unittest.TestCase):
    def test_to_metric(self) -> None:
        self.assertIsInstance(SQLEmExConfig("Accuracy").to_metric(), SQLEmEx)
