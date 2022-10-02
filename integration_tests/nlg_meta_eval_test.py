import os
import unittest

from integration_tests.utils import test_artifacts_path
from explainaboard.metrics.nlg_meta_evaluation import NLGCorrelationConfig

from explainaboard import (
    FileType,
    get_loader_class,
    get_processor_class,
    Source,
    TaskType,
)


class MetaEvalWMTDATest(unittest.TestCase):
    artifact_path = os.path.join(test_artifacts_path, "nlg_meta_evaluation")
    tsv_dataset = os.path.join(artifact_path, "./wmt20-DA/cs-en/data.tsv")
    txt_output = os.path.join(artifact_path, "./wmt20-DA/cs-en/score.txt")

    def test_da_cs_en(self):

        metadata = {
            "task_name": TaskType.nlg_meta_evaluation.value,
            "metric_names": ["SysPearsonCorr"],
            "confidence_alpha": None,
        }
        loader = get_loader_class(TaskType.nlg_meta_evaluation)(
            self.tsv_dataset,
            self.txt_output,
            Source.local_filesystem,
            Source.local_filesystem,
            FileType.tsv,
            FileType.text,
        )
        data = loader.load().samples
        processor = get_processor_class(TaskType.nlg_meta_evaluation)()

        sys_info = processor.process(metadata, data)

        self.assertGreater(len(sys_info.results.analyses), 0)
        self.assertGreater(len(sys_info.results.overall), 0)


class MetaEvalNLGTest(unittest.TestCase):
    true_data = [[1, 2, 3, 4, 5], [2, 1, 4, 5, 2], [5, 4, 3, 2, 1]]
    pred_data = [[2, 1, 3, 4, 5], [2, 4, 5, 5, 2], [5, 3, 4, 2, 1]]

    def test_sample_level_spearmanr(self):

        nlg_corr_config = NLGCorrelationConfig(
            name = "NLGCorrelation",
            level = "sample",
            func_name = "spearmanr"
        )
        corr_metric = nlg_corr_config.to_metric()
        stats = corr_metric.calc_stats_from_data(self.true_data, self.pred_data)
        stats_arr = corr_metric.aggregate_stats(stats)
        val = nlg_corr_config.to_metric().calc_metric_from_aggregate_single(stats_arr)
        self.assertAlmostEqual(val, 0.8162952, 3)

    def test_sample_level_kendalltau(self):

        nlg_corr_config = NLGCorrelationConfig(
            name = "NLGCorrelation",
            level = "sample",
            func_name = "kendalltau"
        )
        corr_metric = nlg_corr_config.to_metric()
        stats = corr_metric.calc_stats_from_data(self.true_data, self.pred_data)
        stats_arr = corr_metric.aggregate_stats(stats)
        val = nlg_corr_config.to_metric().calc_metric_from_aggregate_single(stats_arr)
        self.assertAlmostEqual(val, 0.69046817, 3)

    def test_sample_level_pearsonr(self):
        nlg_corr_config = NLGCorrelationConfig(
            name = "NLGCorrelation",
            level = "sample",
            func_name = "pearsonr"
        )
        corr_metric = nlg_corr_config.to_metric()
        stats = corr_metric.calc_stats_from_data(self.true_data, self.pred_data)
        stats_arr = corr_metric.aggregate_stats(stats)
        val = nlg_corr_config.to_metric().calc_metric_from_aggregate_single(stats_arr)
        self.assertAlmostEqual(val, 0.820707397, 3)


    def test_system_level_spearmanr(self):

        nlg_corr_config = NLGCorrelationConfig(
            name = "NLGCorrelation",
            level = "system",
            func_name = "spearmanr"
        )
        corr_metric = nlg_corr_config.to_metric()
        stats = corr_metric.calc_stats_from_data(self.true_data, self.pred_data)
        stats_arr = corr_metric.aggregate_stats(stats)
        val = corr_metric.calc_metric_from_aggregate(stats_arr)
        self.assertAlmostEqual(val, 0.815789, 3)


    def test_system_level_kendalltau(self):

        nlg_corr_config = NLGCorrelationConfig(
            name = "NLGCorrelation",
            level = "system",
            func_name = "kendalltau"
        )
        corr_metric = nlg_corr_config.to_metric()
        stats = corr_metric.calc_stats_from_data(self.true_data, self.pred_data)
        stats_arr = corr_metric.aggregate_stats(stats)
        val = corr_metric.calc_metric_from_aggregate(stats_arr)
        self.assertAlmostEqual(val, 0.66666, 3)



    def test_dataset_level_spearmanr(self):
        true_data = [[1], [2], [3], [4], [5]]
        pred_data = [[1], [2], [3], [4], [5]]

        nlg_corr_config = NLGCorrelationConfig(
            name = "NLGCorrelation",
            level = "dataset",
            func_name = "spearmanr"
        )
        corr_metric = nlg_corr_config.to_metric()
        stats = corr_metric.calc_stats_from_data(true_data, pred_data)
        stats_arr = corr_metric.aggregate_stats(stats)
        val = corr_metric.calc_metric_from_aggregate(stats_arr)
        self.assertAlmostEqual(val, 1., 3)




class MetaEvalNLGCITest(unittest.TestCase):
    true_data = [[1, 2, 3, 4, 5], [2, 1, 4, 5, 2], [5, 4, 3, 2, 1]]
    pred_data = [[2, 1, 3, 4, 5], [2, 4, 5, 5, 2], [5, 3, 4, 2, 1]]

    def test_sample_level_spearmanr_bootstrap(self):

        nlg_corr_config = NLGCorrelationConfig(
            name = "NLGCorrelation",
            level = "sample",
            func_name = "spearmanr"
        )
        corr_metric = nlg_corr_config.to_metric()
        stats = corr_metric.calc_stats_from_data(self.true_data, self.pred_data)
        stats_arr = corr_metric.aggregate_stats(stats)
        val = corr_metric.calc_metric_from_aggregate_single(stats_arr)
        self.assertAlmostEqual(val, 0.8162952, 3)

        ci = corr_metric.calc_confidence_interval(stats, 0.05)
        self.assertGreater(val, ci[0])
        self.assertGreater(ci[1], val)


    def test_system_level_spearmanr_bootstrap(self):

        nlg_corr_config = NLGCorrelationConfig(
            name = "NLGCorrelation",
            level = "system",
            func_name = "spearmanr"
        )
        corr_metric = nlg_corr_config.to_metric()
        stats = corr_metric.calc_stats_from_data(self.true_data, self.pred_data)
        stats_arr = corr_metric.aggregate_stats(stats)
        val = corr_metric.calc_metric_from_aggregate(stats_arr)
        self.assertAlmostEqual(val, 0.815789, 3)

        ci = corr_metric.calc_confidence_interval(stats, 0.05)
        self.assertGreater(val, ci[0])
        self.assertGreater(ci[1], val)