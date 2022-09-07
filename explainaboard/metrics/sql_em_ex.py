from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, cast

import numpy as np

from explainaboard.metrics.metric import (
    Metric,
    MetricConfig,
    MetricStats,
    SimpleMetricStats,
)
from explainaboard.metrics.registry import metric_config_registry
from explainaboard.metrics.auxiliary.sql_evaluation.sql_em_ex_auxiliary import sql_evaluate


@dataclass
@metric_config_registry.register("SQLEmConfig")
class SQLEmConfig(MetricConfig):
    db_dir: str = ''
    table_path: str = ''
    etype: str = 'match'

    def to_metric(self):
        return SQLEm(self)


class SQLEm(Metric):
    """
    Calculate exact set match (Em), where score is 1 iff the prediction SQL equals the ground
    truth on their SQL string match, respectively
    """

    def calc_stats_from_data(
            self, true_data: list, pred_data: list, config: Optional[MetricConfig] = None
    ) -> MetricStats:
        config = cast(SQLEmConfig, config or self.config)
        em_list = sql_evaluate(true_data, pred_data, config)
        return SimpleMetricStats(np.array(em_list))


@dataclass
@metric_config_registry.register("SQLExConfig")
class SQLExConfig(MetricConfig):
    db_dir: str = ''
    table_path: str = ''
    etype: str = 'exec'

    def to_metric(self):
        return SQLEx(self)


class SQLEx(Metric):
    """
    Calculate exact set match (Em) and execuation (Ex) accuracy, where score is 1 iff the prediction SQL equals the ground
    truth on their SQL string match and execuation, respectively
    """

    def calc_stats_from_data(
            self, true_data: list, pred_data: list, config: Optional[MetricConfig] = None
    ) -> MetricStats:
        config = cast(SQLExConfig, config or self.config)
        ex_list = sql_evaluate(true_data, pred_data, config)
        return SimpleMetricStats(np.array(ex_list))
