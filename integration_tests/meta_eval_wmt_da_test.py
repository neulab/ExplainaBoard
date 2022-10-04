import os
import unittest

from integration_tests.utils import test_artifacts_path

from explainaboard import (
    FileType,
    get_loader_class,
    get_processor_class,
    Source,
    TaskType,
)
from explainaboard.metrics.nlg_meta_evaluation import (
    CorrelationNLG,
    CorrelationNLGConfig,
)
from explainaboard.utils.typing_utils import narrow, unwrap


class MetaEvalWMTDATest(unittest.TestCase):
    artifact_path = os.path.join(test_artifacts_path, "nlg_meta_evaluation")
    tsv_dataset = os.path.join(artifact_path, "./wmt20-DA/cs-en/data.tsv")
    txt_output = os.path.join(artifact_path, "./wmt20-DA/cs-en/score.txt")

    def test_da_cs_en(self):

        metadata = {
            "task_name": TaskType.meta_evaluation_wmt_da.value,
            "metric_names": ["SysPearsonCorr"],
            "confidence_alpha": None,
        }
        loader = get_loader_class(TaskType.meta_evaluation_wmt_da)(
            self.tsv_dataset,
            self.txt_output,
            Source.local_filesystem,
            Source.local_filesystem,
            FileType.tsv,
            FileType.text,
        )
        data = loader.load().samples
        processor = get_processor_class(TaskType.meta_evaluation_wmt_da)()

        sys_info = processor.process(metadata, data)
        self.assertGreater(len(sys_info.results.analyses), 0)
        self.assertGreater(len(sys_info.results.overall), 0)
        self.assertAlmostEqual(
            sys_info.results.overall[0]["SegKtauCorr"].value, -0.0169, 3
        )


class MetaEvalNLGInvalidValueTest(unittest.TestCase):

    true_data = [[1, 2, 3, 4, 5], [2, 1, 4, 5, 2], [5, 4, 3, 2, 1]]
    pred_data = [[2, 1, 3, 4, 5], [2, 4, 5, 5, 2], [5, 3, 4, 2, 1]]

    def test_illegal_correlation_type(self) -> None:

        with self.assertRaisesRegex(
            ValueError, r"^The correlation " r"function illegal hasn't been supported"
        ):
            nlg_corr_config = CorrelationNLGConfig(
                group_by="sample", correlation_type="illegal"
            )
            corr_metric = narrow(CorrelationNLG, nlg_corr_config.to_metric())
            stats = corr_metric.calc_stats_from_data(self.true_data, self.pred_data)
            stats_arr = corr_metric.aggregate_stats(stats)
            val = corr_metric._calc_metric_from_aggregate_single(stats_arr)
            self.assertAlmostEqual(val, 0.8162952, 3)

    def test_illegal_group_type(self) -> None:

        with self.assertRaisesRegex(
            ValueError, r"^group_by with the value illegal hasn't been supported."
        ):

            nlg_corr_config = CorrelationNLGConfig(
                group_by="illegal", correlation_type="spearmanr"
            )
            corr_metric = narrow(CorrelationNLG, nlg_corr_config.to_metric())
            stats = corr_metric.calc_stats_from_data(self.true_data, self.pred_data)
            stats_arr = corr_metric.aggregate_stats(stats)
            val = corr_metric._calc_metric_from_aggregate_single(stats_arr)
            self.assertAlmostEqual(val, 0.8162952, 3)


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
        corr_metric = narrow(CorrelationNLG, nlg_corr_config.to_metric())
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
        corr_metric = narrow(CorrelationNLG, nlg_corr_config.to_metric())
        stats = corr_metric.calc_stats_from_data(self.true_data, self.pred_data)
        stats_arr = corr_metric.aggregate_stats(stats)
        val = corr_metric.calc_metric_from_aggregate(stats_arr)
        self.assertAlmostEqual(val, 0.815789, 3)

        ci = unwrap(corr_metric.calc_confidence_interval(stats, 0.05))
        self.assertAlmostEqual(ci[0], 0.5642, 2)
        self.assertAlmostEqual(ci[1], 0.9746, 2)
