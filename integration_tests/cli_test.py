from __future__ import annotations

import logging
from pathlib import Path
import tempfile
import unittest
from unittest import TestCase
from unittest.mock import patch

from datalabs import set_progress_bar_enabled
from integration_tests.utils import OPTIONAL_TEST_SUITES, top_path

import explainaboard.explainaboard_main
from explainaboard.utils.cache_api import cache_online_file
from explainaboard.utils.logging import get_logger
import explainaboard.visualizers.draw_charts


class CLITest(TestCase):
    """TODO: these tests only tests if they run. After the main script
    has been refactored, we can make these tests more useful"""

    def setUp(self):
        # To disable non-critical logging.
        for name in [None, "report"]:
            get_logger(name).setLevel(logging.WARNING)
        # To disable progress bar when downloading datasets using datalabs.
        set_progress_bar_enabled(False)

    def test_textclass_datalab(self):
        args = [
            "explainaboard.explainaboard_main",
            "--task",
            "text-classification",
            "--system-outputs",
            f"{top_path}/data/system_outputs/sst2/sst2-lstm-output.txt",
            "--dataset",
            "sst2",
            "--report-json",
            "/dev/null",
        ]
        with patch("sys.argv", args):
            explainaboard.explainaboard_main.main()

    def test_textclass_datalab_pairwise(self):
        args = [
            "explainaboard.explainaboard_main",
            "--task",
            "text-classification",
            "--system-outputs",
            f"{top_path}/data/system_outputs/sst2/sst2-lstm-output.txt",
            f"{top_path}/data/system_outputs/sst2/sst2-cnn-output.txt",
            "--dataset",
            "sst2",
            "--report-json",
            "/dev/null",
        ]
        with patch("sys.argv", args):
            explainaboard.explainaboard_main.main()

    def test_textclass_viz(self):
        with tempfile.TemporaryDirectory() as tempdir:
            td = Path(tempdir)
            reports_dir = td / "reports"
            figures_dir = td / "figures"
            reports_dir.mkdir(parents=True)
            figures_dir.mkdir(parents=True)
            for sysname in ("lstm", "cnn"):
                args = [
                    "explainaboard.explainaboard_main",
                    "--task",
                    "text-classification",
                    "--system-outputs",
                    f"{top_path}/data/system_outputs/sst2/sst2-{sysname}-output.txt",
                    "--dataset",
                    "sst2",
                    "--report-json",
                    str(reports_dir / f"sst2-{sysname}-output.json"),
                ]
                with patch("sys.argv", args):
                    explainaboard.explainaboard_main.main()
            args = [
                "explainaboard.visualizers.draw_hist",
                "--reports",
                str(reports_dir / "sst2-lstm-output.json"),
                str(reports_dir / "sst2-cnn-output.json"),
                "--output-dir",
                str(figures_dir),
            ]
            with patch("sys.argv", args):
                explainaboard.visualizers.draw_charts.main()

    def test_textclass_custom(self):
        args = [
            "explainaboard.explainaboard_main",
            "--task",
            "text-classification",
            "--system-outputs",
            f"{top_path}/data/system_outputs/sst2/sst2-lstm-output.txt",
            "--custom-dataset-paths",
            f"{top_path}/data/system_outputs/sst2/sst2-dataset.tsv",
            "--report-json",
            "/dev/null",
            "--skip-failed-analyses",
        ]
        with patch("sys.argv", args):
            explainaboard.explainaboard_main.main()

    def test_tabreg_custom(self):
        args = [
            "explainaboard.explainaboard_main",
            "--task",
            "tabular-regression",
            "--system-outputs",
            f"{top_path}/data/system_outputs/sst2_tabreg/sst2-tabreg-lstm-output.txt",
            "--custom-dataset-paths",
            f"{top_path}/data/system_outputs/sst2_tabreg/sst2-tabreg-dataset.json",
            "--report-json",
            "/dev/null",
        ]
        with patch("sys.argv", args):
            explainaboard.explainaboard_main.main()

    def test_tabclass_custom(self):
        args = [
            "explainaboard.explainaboard_main",
            "--task",
            "tabular-classification",
            "--system-outputs",
            f"{top_path}/data/system_outputs/sst2/sst2-lstm-output.txt",
            "--custom-dataset-paths",
            f"{top_path}/data/system_outputs/sst2_tabclass/sst2-tabclass-dataset.json",
            "--report-json",
            "/dev/null",
        ]
        with patch("sys.argv", args):
            explainaboard.explainaboard_main.main()

    @unittest.skipUnless("cli_all" in OPTIONAL_TEST_SUITES, reason="time consuming")
    def test_textpair_datalab(self):
        args = [
            "explainaboard.explainaboard_main",
            "--task",
            "text-pair-classification",
            "--system-outputs",
            f"{top_path}/data/system_outputs/snli/snli-roberta-output.txt",
            "--dataset",
            "snli",
            "--report-json",
            "/dev/null",
        ]
        with patch("sys.argv", args):
            explainaboard.explainaboard_main.main()

    def test_textpair_custom(self):
        args = [
            "explainaboard.explainaboard_main",
            "--task",
            "text-pair-classification",
            "--system-outputs",
            f"{top_path}/data/system_outputs/snli/snli-roberta-output.txt",
            "--custom-dataset-paths",
            f"{top_path}/data/system_outputs/snli/snli-dataset.tsv",
            "--report-json",
            "/dev/null",
            "--skip-failed-analyses",
        ]
        with patch("sys.argv", args):
            explainaboard.explainaboard_main.main()

    def test_summ_custom(self):
        args = [
            "explainaboard.explainaboard_main",
            "--task",
            "summarization",
            "--custom-dataset-paths",
            f"{top_path}/data/system_outputs/cnndm/cnndm_mini-dataset.tsv",
            "--system-outputs",
            f"{top_path}/data/system_outputs/cnndm/cnndm_mini-bart-output.txt",
            "--metrics",
            "rouge2",
            "chrf",
            "--report-json",
            "/dev/null",
            "--skip-failed-analyses",
        ]
        with patch("sys.argv", args):
            explainaboard.explainaboard_main.main()

    @unittest.skipUnless("cli_all" in OPTIONAL_TEST_SUITES, reason="time consuming")
    def test_summ_datalab(self):
        filename = cache_online_file(
            "https://storage.googleapis.com/inspired-public-data/"
            "explainaboard/task_data/summarization/cnndm-bart-output.txt",
            "explainaboard/task_data/summarization/cnndm-bart-output.txt",
        )
        args = [
            "explainaboard.explainaboard_main",
            "--task",
            "summarization",
            "--dataset",
            "cnn_dailymail",
            "--system-outputs",
            filename,
            "--metrics",
            "rouge2",
            "--report-json",
            "/dev/null",
        ]
        with patch("sys.argv", args):
            explainaboard.explainaboard_main.main()

    def test_mt_custom(self):
        args = [
            "explainaboard.explainaboard_main",
            "--task",
            "machine-translation",
            "--custom-dataset-paths",
            f"{top_path}/data/system_outputs/ted_multi/ted_multi_slk_eng-dataset.tsv",
            "--system-outputs",
            f"{top_path}/data/system_outputs/ted_multi/ted_multi_slk_eng-nmt-output.txt",  # noqa
            "--metrics",
            "bleu",
            "--report-json",
            "/dev/null",
            "--skip-failed-analyses",
        ]
        with patch("sys.argv", args):
            explainaboard.explainaboard_main.main()

    def test_codegen_custom(self):
        args = [
            "explainaboard.explainaboard_main",
            "--task",
            "machine-translation",
            "--custom-dataset-file-type",
            "json",
            "--custom-dataset-paths",
            f"{top_path}/data/system_outputs/conala/conala-dataset.json",
            "--output-file-type",
            "json",
            "--system-outputs",
            f"{top_path}/data/system_outputs/conala/conala-baseline-output.json",
            "--report-json",
            "report.json",
            "--skip-failed-analyses",
        ]
        with patch("sys.argv", args):
            explainaboard.explainaboard_main.main()

    def test_codegen_datalab(self):
        args = [
            "explainaboard.explainaboard_main",
            "--task",
            "machine-translation",
            "--dataset",
            "conala",
            "--output-file-type",
            "json",
            "--system-outputs",
            f"{top_path}/data/system_outputs/conala/conala-baseline-output.json",
            "--report-json",
            "report.json",
        ]
        with patch("sys.argv", args):
            explainaboard.explainaboard_main.main()

    def test_lm_custom(self):
        args = [
            "explainaboard.explainaboard_main",
            "--task",
            "language-modeling",
            "--custom-dataset-paths",
            f"{top_path}/data/system_outputs/wikitext/wikitext-dataset.txt",
            "--system-outputs",
            f"{top_path}/data/system_outputs/wikitext/wikitext-sys1-output.txt",
            "--report-json",
            "/dev/null",
            "--skip-failed-analyses",
        ]
        with patch("sys.argv", args):
            explainaboard.explainaboard_main.main()

    def test_ner_datalab(self):
        args = [
            "explainaboard.explainaboard_main",
            "--task",
            "named-entity-recognition",
            "--dataset",
            "conll2003",
            "--sub-dataset",
            "ner",
            "--system-outputs",
            f"{top_path}/data/system_outputs/conll2003/conll2003-elmo-output.conll",
            "--report-json",
            "/dev/null",
        ]
        with patch("sys.argv", args):
            explainaboard.explainaboard_main.main()

    def test_ner_custom(self):
        args = [
            "explainaboard.explainaboard_main",
            "--task",
            "named-entity-recognition",
            "--custom-dataset-paths",
            f"{top_path}/data/system_outputs/conll2003/conll2003-dataset.conll",
            "--system-outputs",
            f"{top_path}/data/system_outputs/conll2003/conll2003-elmo-output.conll",
            "--report-json",
            "/dev/null",
            "--skip-failed-analyses",
        ]
        with patch("sys.argv", args):
            explainaboard.explainaboard_main.main()

    def test_multichoiceqa_datalab(self):
        args = [
            "explainaboard.explainaboard_main",
            "--task",
            "qa-multiple-choice",
            "--dataset",
            "fig_qa",
            "--split",
            "validation",
            "--system-outputs",
            f"{top_path}/data/system_outputs/fig_qa/fig_qa-gptneo-output.json",  # noqa
            "--report-json",
            "/dev/null",
        ]
        with patch("sys.argv", args):
            explainaboard.explainaboard_main.main()

    def test_multichoiceqa_custom(self):
        args = [
            "explainaboard.explainaboard_main",
            "--task",
            "qa-multiple-choice",
            "--custom-dataset-paths",
            f"{top_path}/data/system_outputs/fig_qa/fig_qa-dataset.json",
            "--system-outputs",
            f"{top_path}/data/system_outputs/fig_qa/fig_qa-gptneo-output.json",  # noqa
            "--report-json",
            "/dev/null",
            "--skip-failed-analyses",
        ]
        with patch("sys.argv", args):
            explainaboard.explainaboard_main.main()

    def test_extractiveqa_custom(self):
        args = [
            "explainaboard.explainaboard_main",
            "--task",
            "qa-extractive",
            "--custom-dataset-paths",
            f"{top_path}/data/system_outputs/squad/squad_mini-dataset.json",
            "--system-outputs",
            f"{top_path}/data/system_outputs/squad/squad_mini-example-output.json",
            "--report-json",
            "/dev/null",
            "--skip-failed-analyses",
        ]
        with patch("sys.argv", args):
            explainaboard.explainaboard_main.main()

    @unittest.skip(
        reason="to be fixed in future PR: "
        "https://github.com/neulab/ExplainaBoard/issues/247"
    )
    def test_kglinktail_datalab(self):
        args = [
            "explainaboard.explainaboard_main",
            "--task",
            "kg-link-tail-prediction",
            "--dataset",
            "fb15k_237",
            "--system-outputs",
            f"{top_path}/data/system_outputs/fb15k-237/test-kg-prediction-no-user-defined.json",  # noqa
            "--report-json",
            "/dev/null",
        ]
        with patch("sys.argv", args):
            explainaboard.explainaboard_main.main()

    def test_kglinktail_custom(self):
        args = [
            "explainaboard.explainaboard_main",
            "--task",
            "kg-link-tail-prediction",
            "--custom-dataset-paths",
            f"{top_path}/data/system_outputs/fb15k-237/test-kg-prediction-no-user-defined.json",  # noqa
            "--system-outputs",
            f"{top_path}/data/system_outputs/fb15k-237/test-kg-prediction-no-user-defined.json",  # noqa
            "--report-json",
            "/dev/null",
            "--skip-failed-analyses",
        ]
        with patch("sys.argv", args):
            explainaboard.explainaboard_main.main()

    def test_absa_custom(self):
        args = [
            "explainaboard.explainaboard_main",
            "--task",
            "aspect-based-sentiment-classification",
            "--custom-dataset-paths",
            f"{top_path}/data/system_outputs/absa/absa-dataset.tsv",
            "--system-outputs",
            f"{top_path}/data/system_outputs/absa/absa-example-output.txt",
            "--report-json",
            "/dev/null",
        ]
        with patch("sys.argv", args):
            explainaboard.explainaboard_main.main()
