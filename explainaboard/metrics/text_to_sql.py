"""Evaluation metrics for the text-to-sql task.

Metrics contains exact set match accuracy and execution accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast, Optional

import numpy as np

from explainaboard.metrics.metric import (
    Metric,
    MetricConfig,
    MetricStats,
    SimpleMetricStats,
)
from explainaboard.serialization import common_registry
from explainaboard.third_party.text_to_sql_test_suit_eval.evaluation import evaluate


@dataclass
@common_registry.register("SQLExactSetMatchConfig")
class SQLExactSetMatchConfig(MetricConfig):
    """Configuration for SQLExactSetMatch.

    Args:
        db_dir: the path to database folder.
        table_path: the path to table schema file.
        etype: the evaluation type, "match" or "exec".
    """

    db_dir: str = ""
    table_path: str = ""
    etype: str = "match"

    def to_metric(self):
        """See MetricConfig.to_metric."""
        return SQLExactSetMatch(self)


class SQLExactSetMatch(Metric):
    """Calculate exact set match (Em) accuracy.

    The score is 1 iff the prediction SQL equals the ground
    truth on their SQL string match.
    """

    def calc_stats_from_data(
        self, true_data: list, pred_data: list, config: Optional[MetricConfig] = None
    ) -> MetricStats:
        """See Metric.calc_stats_from_data.

        Args:
          true_data: a list of gold sqls.
          pred_data: a list of predicted sqls.

        Returns:
          See Metric.calc_stats_from_data.
        """
        config = cast(SQLExactSetMatchConfig, config or self.config)
        config_dict = {
            "db_dir": config.db_dir,
            "table_path": config.table_path,
            "etype": config.etype,
        }
        em_list = evaluate(true_data, pred_data, config_dict)
        return SimpleMetricStats(np.array(em_list))


@dataclass
@common_registry.register("SQLExecutionConfig")
class SQLExecutionConfig(MetricConfig):
    """Configuration for SQLExecution.

    Args:
        db_dir: the path to database folder.
        table_path: the path to table schema file.
        etype: the evaluation type, "match" or "exec".
    """

    db_dir: str = ""
    table_path: str = ""
    etype: str = "exec"

    def to_metric(self):
        """See MetricConfig.to_metric."""
        return SQLExecution(self)


class SQLExecution(Metric):
    """Calculate execution (Ex) accuracy.

    The score is 1 iff the prediction SQL generates the same execution results
    as the groundtruth SQL.
    """

    def calc_stats_from_data(
        self, true_data: list, pred_data: list, config: Optional[MetricConfig] = None
    ) -> MetricStats:
        """See Metric.calc_stats_from_data.

        Args:
          true_data: a list of gold sqls.
          pred_data: a list of predicted sqls.

        Returns:
          See Metric.calc_stats_from_data.
        """
        config = cast(SQLExecutionConfig, config or self.config)
        config_dict = {
            "db_dir": config.db_dir,
            "table_path": config.table_path,
            "etype": config.etype,
        }
        ex_list = evaluate(true_data, pred_data, config_dict)
        return SimpleMetricStats(np.array(ex_list))
