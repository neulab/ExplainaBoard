"""Tests for explainaboard.third_party.text_to_sql_test_suit_eval.sql_evaluation."""

from __future__ import annotations

import unittest

from explainaboard.third_party.text_to_sql_test_suit_eval.evaluation import evaluate


class SQL_Evaluation_Test(unittest.TestCase):
    def test_sql_evaluation_metric(self) -> None:
        true = [
            ["select distinct country from singer where age > 20", "concert_singer"],
            ["select distinct country from singer where age > 20", "concert_singer"],
            ["select distinct country from singer where age > 20", "concert_singer"],
        ]
        pred = [
            ["select distinct country from singer where age > 20", "concert_singer"],
            ["select distinct country from singer where age > 25", "concert_singer"],
            ["select distinct country from singer where age = 20", "concert_singer"],
        ]
        config = {
            "db_dir": "https://expressai-xlab.s3.amazonaws.com/large_data/database",
            "table_path": "https://expressai-xlab.s3.amazonaws.com/"
            "large_data/table/tables.json",
            "etype": "exec",
        }
        result_raw = evaluate(true, pred, config)
        result = sum(result_raw) / len(result_raw)
        self.assertAlmostEqual(result, 1.0 / 3.0)

        config = {
            "db_dir": "https://expressai-xlab.s3.amazonaws.com/large_data/database",
            "table_path": "https://expressai-xlab.s3.amazonaws.com/"
            "large_data/table/tables.json",
            "etype": "match",
        }
        result_raw = evaluate(true, pred, config)
        result = sum(result_raw) / len(result_raw)
        self.assertAlmostEqual(result, 2.0 / 3.0)
