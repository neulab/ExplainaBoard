from __future__ import annotations

import unittest

import numpy as np

from explainaboard.metrics.meta_evaluation import CorrelationNLG, CorrelationNLGConfig
from explainaboard.utils.typing_utils import narrow, unwrap


class MetaEvalNLGInvalidValueTest(unittest.TestCase):

    true_data = [[1, 2, 3, 4, 5], [2, 1, 4, 5, 2], [5, 4, 3, 2, 1]]
    pred_data = [[2, 1, 3, 4, 5], [2, 4, 5, 5, 2], [5, 3, 4, 2, 1]]

    def test_illegal_correlation_type_calc_stats_from_data(self) -> None:

        nlg_corr_config = CorrelationNLGConfig(
            group_by="sample", correlation_type="illegal"
        )

        corr_metric = narrow(CorrelationNLG, nlg_corr_config.to_metric())
        with self.assertRaisesRegex(ValueError, r"^The correlation function"):
            corr_metric.calc_stats_from_data(self.true_data, self.pred_data)

    def test_illegal_correlation_type_aggregate_stats(self) -> None:
        nlg_corr_config = CorrelationNLGConfig(
            group_by="sample", correlation_type="illegal"
        )
        corr_metric = narrow(CorrelationNLG, nlg_corr_config.to_metric())
        stats_arr = np.zeros((3, 1))
        with self.assertRaisesRegex(ValueError, r"^The correlation function"):
            corr_metric._calc_metric_from_aggregate_single(stats_arr)

    def test_illegal_group_type_calc_stats_from_data(self) -> None:

        nlg_corr_config = CorrelationNLGConfig(
            group_by="illegal", correlation_type="spearmanr"
        )

        corr_metric = narrow(CorrelationNLG, nlg_corr_config.to_metric())
        with self.assertRaisesRegex(
            ValueError, r"^group_by with the value illegal" r" hasn't been supported."
        ):
            corr_metric.calc_stats_from_data(self.true_data, self.pred_data)

    def test_illegal_group_type_aggregate_stats(self) -> None:
        nlg_corr_config = CorrelationNLGConfig(
            group_by="illegal", correlation_type="spearmanr"
        )
        corr_metric = narrow(CorrelationNLG, nlg_corr_config.to_metric())
        stats_arr = np.zeros((3, 1))
        with self.assertRaisesRegex(
            ValueError, r"^group_by with the value illegal" r" hasn't been supported."
        ):
            corr_metric._calc_metric_from_aggregate_single(stats_arr)


class MetaEvalNLGTest(unittest.TestCase):
    true_data = [[1, 2, 3, 4, 5], [2, 1, 4, 5, 2], [5, 4, 3, 2, 1]]
    pred_data = [[2, 1, 3, 4, 5], [2, 4, 5, 5, 2], [5, 3, 4, 2, 1]]

    def test_sample_level_spearmanr(self) -> None:

        nlg_corr_config = CorrelationNLGConfig(
            group_by="sample", correlation_type="spearmanr"
        )

        corr_metric = narrow(CorrelationNLG, nlg_corr_config.to_metric())
        stats = corr_metric.calc_stats_from_data(self.true_data, self.pred_data)
        stats_arr = corr_metric.aggregate_stats(stats)
        val = corr_metric._calc_metric_from_aggregate_single(stats_arr)
        self.assertAlmostEqual(val, 0.8162952, 3)

    def test_sample_level_kendalltau(self) -> None:

        nlg_corr_config = CorrelationNLGConfig(
            group_by="sample", correlation_type="kendalltau"
        )
        corr_metric = narrow(CorrelationNLG, nlg_corr_config.to_metric())
        stats = corr_metric.calc_stats_from_data(self.true_data, self.pred_data)
        stats_arr = corr_metric.aggregate_stats(stats)
        val = corr_metric._calc_metric_from_aggregate_single(stats_arr)
        self.assertAlmostEqual(val, 0.69046817, 3)

    def test_sample_level_pearsonr(self) -> None:

        nlg_corr_config = CorrelationNLGConfig(
            group_by="sample", correlation_type="pearsonr"
        )
        corr_metric = narrow(CorrelationNLG, nlg_corr_config.to_metric())
        stats = corr_metric.calc_stats_from_data(self.true_data, self.pred_data)
        stats_arr = corr_metric.aggregate_stats(stats)
        val = corr_metric._calc_metric_from_aggregate_single(stats_arr)
        self.assertAlmostEqual(val, 0.820707397, 3)

    def test_system_level_spearmanr(self) -> None:

        nlg_corr_config = CorrelationNLGConfig(
            group_by="system", correlation_type="spearmanr"
        )
        corr_metric = narrow(CorrelationNLG, nlg_corr_config.to_metric())
        stats = corr_metric.calc_stats_from_data(self.true_data, self.pred_data)
        stats_arr = corr_metric.aggregate_stats(stats)
        val = corr_metric.calc_metric_from_aggregate(stats_arr)
        self.assertAlmostEqual(val, 0.815789, 3)

    def test_system_level_kendalltau(self) -> None:

        nlg_corr_config = CorrelationNLGConfig(
            group_by="system", correlation_type="kendalltau"
        )
        corr_metric = narrow(CorrelationNLG, nlg_corr_config.to_metric())
        stats = corr_metric.calc_stats_from_data(self.true_data, self.pred_data)
        stats_arr = corr_metric.aggregate_stats(stats)
        val = corr_metric.calc_metric_from_aggregate(stats_arr)
        self.assertAlmostEqual(val, 0.66666, 3)

    def test_dataset_level_spearmanr(self) -> None:

        true_data = [[1], [2], [3], [4], [5]]
        pred_data = [[1], [2], [3], [4], [5]]

        nlg_corr_config = CorrelationNLGConfig(
            group_by="dataset", correlation_type="spearmanr"
        )
        corr_metric = narrow(CorrelationNLG, nlg_corr_config.to_metric())
        stats = corr_metric.calc_stats_from_data(true_data, pred_data)
        stats_arr = corr_metric.aggregate_stats(stats)
        val = corr_metric.calc_metric_from_aggregate(stats_arr)
        self.assertAlmostEqual(val, 1.0, 3)


class MetaEvalNLGCITest(unittest.TestCase):

    true_data = [[1, 2, 3, 4, 5], [2, 1, 4, 5, 2], [5, 4, 3, 2, 1]]
    pred_data = [[2, 1, 3, 4, 5], [2, 4, 5, 5, 2], [5, 3, 4, 2, 1]]

    def test_sample_level_spearmanr_bootstrap(self) -> None:

        nlg_corr_config = CorrelationNLGConfig(
            group_by="sample", correlation_type="spearmanr"
        )
        corr_metric = CorrelationNLG(
            nlg_corr_config, seed=np.random.SeedSequence(12345)
        )
        stats = corr_metric.calc_stats_from_data(self.true_data, self.pred_data)
        stats_arr = corr_metric.aggregate_stats(stats)
        val = corr_metric._calc_metric_from_aggregate_single(stats_arr)
        self.assertAlmostEqual(val, 0.8162952, 3)

        ci = unwrap(corr_metric.calc_confidence_interval(stats, 0.05))
        self.assertAlmostEqual(ci[0], 0.6488, 2)
        self.assertAlmostEqual(ci[1], 0.8999, 2)

    def test_system_level_spearmanr_bootstrap(self) -> None:

        nlg_corr_config = CorrelationNLGConfig(
            group_by="system", correlation_type="spearmanr"
        )
        corr_metric = CorrelationNLG(
            nlg_corr_config, seed=np.random.SeedSequence(12345)
        )
        stats = corr_metric.calc_stats_from_data(self.true_data, self.pred_data)
        stats_arr = corr_metric.aggregate_stats(stats)
        val = corr_metric.calc_metric_from_aggregate(stats_arr)
        self.assertAlmostEqual(val, 0.815789, 3)

        ci = unwrap(corr_metric.calc_confidence_interval(stats, 0.05))
        self.assertAlmostEqual(ci[0], 0.5642, 2)
        self.assertAlmostEqual(ci[1], 0.9746, 2)

    def test_dataset_level_spearmanr_bootstrap(self) -> None:

        true_data = [[1], [2], [3], [4], [5]]
        pred_data = [[1], [2], [3], [4], [5]]

        nlg_corr_config = CorrelationNLGConfig(
            group_by="dataset", correlation_type="spearmanr"
        )
        corr_metric = CorrelationNLG(
            nlg_corr_config, seed=np.random.SeedSequence(12345)
        )
        stats = corr_metric.calc_stats_from_data(true_data, pred_data)
        stats_arr = corr_metric.aggregate_stats(stats)
        val = corr_metric.calc_metric_from_aggregate(stats_arr)
        self.assertAlmostEqual(val, 1, 3)

        ci = unwrap(corr_metric.calc_confidence_interval(stats, 0.05))
        self.assertAlmostEqual(ci[0], 1, 2)
        self.assertAlmostEqual(ci[1], 1, 2)
