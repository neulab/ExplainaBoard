from __future__ import annotations

import os
import unittest

from integration_tests.utils import test_artifacts_path
import numpy as np

from explainaboard import FileType, get_processor_class, Source, TaskType
from explainaboard.loaders.file_loader import DatalabLoaderOption
from explainaboard.loaders.loader_factory import get_loader_class
from explainaboard.metrics.meta_evaluation import CorrelationNLG, CorrelationNLGConfig
from explainaboard.metrics.metric import Score
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


class MetaEvalNLGNewsroomTest(unittest.TestCase):
    """
    Test the NLG metric on newsroom dataset and replicate the reported results from
    the paper: https://arxiv.org/pdf/2106.11520.pdf
    """

    artifact_path = os.path.join(test_artifacts_path, "newsroom")
    predictions_rouge1 = os.path.join(artifact_path, "rouge1_f_predictions.json")
    predictions_bartscore = os.path.join(
        artifact_path, "bart_score_cnn_ref_hypo_predictions.json"
    )

    def test_coherence_rouge1_f(self):
        loader = get_loader_class(TaskType.meta_evaluation_nlg).from_datalab(
            dataset=DatalabLoaderOption("meval_newsroom", "coherence"),
            output_data=self.predictions_rouge1,
            output_source=Source.local_filesystem,
            output_file_type=FileType.json,
        )
        data = loader.load().samples

        metadata = {
            "task_name": TaskType.meta_evaluation_nlg.value,
            "dataset_name": "meval_newsroom",
            "sub_dataset_name": "coherence",
            "metric_names": ["SpearmanSampleLevelCorr"],
        }
        processor = get_processor_class(TaskType.meta_evaluation_nlg)()
        sys_info = processor.process(metadata, data)
        overall_score = (
            sys_info.results.overall["example"]["SpearmanSampleLevelCorr"]
            .get_value(Score, "score")
            .value
        )
        self.assertGreater(len(sys_info.results.analyses), 0)
        self.assertAlmostEqual(
            overall_score,
            0.0946,
            places=3,
        )

    def test_coherence_bartscore(self):
        loader = get_loader_class(TaskType.meta_evaluation_nlg).from_datalab(
            dataset=DatalabLoaderOption("meval_newsroom", "coherence"),
            output_data=self.predictions_bartscore,
            output_source=Source.local_filesystem,
            output_file_type=FileType.json,
        )
        data = loader.load().samples

        metadata = {
            "task_name": TaskType.meta_evaluation_nlg.value,
            "dataset_name": "meval_newsroom",
            "sub_dataset_name": "coherence",
            "metric_names": ["SpearmanSampleLevelCorr"],
        }
        processor = get_processor_class(TaskType.meta_evaluation_nlg)()
        sys_info = processor.process(metadata, data)
        overall_score = (
            sys_info.results.overall["example"]["SpearmanSampleLevelCorr"]
            .get_value(Score, "score")
            .value
        )
        self.assertGreater(len(sys_info.results.analyses), 0)
        self.assertAlmostEqual(
            overall_score,
            0.3157,
            places=3,
        )
