import logging
from pathlib import Path
import unittest
from unittest import TestCase
from unittest.mock import patch

from datalabs import set_progress_bar_enabled
from integration_tests.utils import OPTIONAL_TEST_SUITES, test_output_path, top_path

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
            'explainaboard.explainaboard_main',
            '--task',
            'text-classification',
            '--system_outputs',
            f'{top_path}/data/system_outputs/sst2/sst2-lstm-output.txt',
            '--dataset',
            'sst2',
            '--report_json',
            '/dev/null',
        ]
        with patch('sys.argv', args):
            explainaboard.explainaboard_main.main()

    def test_textclass_datalab_pairwise(self):
        args = [
            'explainaboard.explainaboard_main',
            '--task',
            'text-classification',
            '--system_outputs',
            f'{top_path}/data/system_outputs/sst2/sst2-lstm-output.txt',
            f'{top_path}/data/system_outputs/sst2/sst2-cnn-output.txt',
            '--dataset',
            'sst2',
            '--report_json',
            '/dev/null',
        ]
        with patch('sys.argv', args):
            explainaboard.explainaboard_main.main()

    def test_textclass_viz(self):
        Path(f"{test_output_path}/reports").mkdir(parents=True, exist_ok=True)
        Path(f"{test_output_path}/figures").mkdir(parents=True, exist_ok=True)
        for sysname in ('lstm', 'cnn'):
            args = [
                'explainaboard.explainaboard_main',
                '--task',
                'text-classification',
                '--system_outputs',
                f'{top_path}/data/system_outputs/sst2/sst2-{sysname}-output.txt',
                '--dataset',
                'sst2',
                '--report_json',
                f'{test_output_path}/reports/sst2-{sysname}-output.json',  # noqa
            ]
            with patch('sys.argv', args):
                explainaboard.explainaboard_main.main()
        args = [
            'explainaboard.visualizers.draw_hist',
            '--reports',
            f'{test_output_path}/reports/sst2-lstm-output.json',
            f'{test_output_path}/reports/sst2-cnn-output.json',
            '--output_dir',
            f'{test_output_path}/figures/',
        ]
        with patch('sys.argv', args):
            explainaboard.visualizers.draw_charts.main()

    def test_textclass_custom(self):
        args = [
            'explainaboard.explainaboard_main',
            '--task',
            'text-classification',
            '--system_outputs',
            f'{top_path}/data/system_outputs/sst2/sst2-lstm-output.txt',
            '--custom_dataset_paths',
            f'{top_path}/data/system_outputs/sst2/sst2-dataset.tsv',
            '--report_json',
            '/dev/null',
        ]
        with patch('sys.argv', args):
            explainaboard.explainaboard_main.main()

    def test_tabreg_custom(self):
        args = [
            'explainaboard.explainaboard_main',
            '--task',
            'tabular-regression',
            '--system_outputs',
            f'{top_path}/data/system_outputs/sst2_tabreg/sst2-tabreg-lstm-output.txt',
            '--custom_dataset_paths',
            f'{top_path}/data/system_outputs/sst2_tabreg/sst2-tabreg-dataset.json',
            '--report_json',
            '/dev/null',
        ]
        with patch('sys.argv', args):
            explainaboard.explainaboard_main.main()

    def test_tabclass_custom(self):
        args = [
            'explainaboard.explainaboard_main',
            '--task',
            'tabular-classification',
            '--system_outputs',
            f'{top_path}/data/system_outputs/sst2/sst2-lstm-output.txt',
            '--custom_dataset_paths',
            f'{top_path}/data/system_outputs/sst2_tabclass/sst2-tabclass-dataset.json',
            '--report_json',
            '/dev/null',
        ]
        with patch('sys.argv', args):
            explainaboard.explainaboard_main.main()

    @unittest.skipUnless('cli_all' in OPTIONAL_TEST_SUITES, reason='time consuming')
    def test_textpair_datalab(self):
        args = [
            'explainaboard.explainaboard_main',
            '--task',
            'text-pair-classification',
            '--system_outputs',
            f'{top_path}/data/system_outputs/snli/snli-roberta-output.txt',
            '--dataset',
            'snli',
            '--report_json',
            '/dev/null',
        ]
        with patch('sys.argv', args):
            explainaboard.explainaboard_main.main()

    def test_textpair_custom(self):
        args = [
            'explainaboard.explainaboard_main',
            '--task',
            'text-pair-classification',
            '--system_outputs',
            f'{top_path}/data/system_outputs/snli/snli-roberta-output.txt',
            '--custom_dataset_paths',
            f'{top_path}/data/system_outputs/snli/snli-dataset.tsv',
            '--report_json',
            '/dev/null',
        ]
        with patch('sys.argv', args):
            explainaboard.explainaboard_main.main()

    def test_summ_custom(self):
        args = [
            'explainaboard.explainaboard_main',
            '--task',
            'summarization',
            '--custom_dataset_paths',
            f'{top_path}/data/system_outputs/cnndm/cnndm_mini-dataset.tsv',
            '--system_outputs',
            f'{top_path}/data/system_outputs/cnndm/cnndm_mini-bart-output.txt',
            '--metrics',
            'rouge2',
            'chrf',
            '--report_json',
            '/dev/null',
        ]
        with patch('sys.argv', args):
            explainaboard.explainaboard_main.main()

    @unittest.skipUnless('cli_all' in OPTIONAL_TEST_SUITES, reason='time consuming')
    def test_summ_datalab(self):
        filename = cache_online_file(
            'http://www.phontron.com/download/cnndm-bart-output.txt',
            'tests/cnndm-bart-output.txt',
        )
        args = [
            'explainaboard.explainaboard_main',
            '--task',
            'summarization',
            '--dataset',
            'cnn_dailymail',
            '--system_outputs',
            filename,
            '--metrics',
            'rouge2',
            '--report_json',
            '/dev/null',
        ]
        with patch('sys.argv', args):
            explainaboard.explainaboard_main.main()

    def test_mt_custom(self):
        args = [
            'explainaboard.explainaboard_main',
            '--task',
            'machine-translation',
            '--custom_dataset_paths',
            f'{top_path}/data/system_outputs/ted_multi/ted_multi_slk_eng-dataset.tsv',
            '--system_outputs',
            f'{top_path}/data/system_outputs/ted_multi/ted_multi_slk_eng-nmt-output.txt',  # noqa
            '--metrics',
            'bleu',
            '--report_json',
            '/dev/null',
        ]
        with patch('sys.argv', args):
            explainaboard.explainaboard_main.main()

    def test_codegen_custom(self):
        args = [
            'explainaboard.explainaboard_main',
            "--task",
            "machine-translation",
            "--custom_dataset_file_type",
            "json",
            "--custom_dataset_paths",
            f"{top_path}/data/system_outputs/conala/conala-dataset.json",
            "--output_file_type",
            "json",
            "--system_outputs",
            f"{top_path}/data/system_outputs/conala/conala-baseline-output.json",
            "--report_json",
            "report.json",
        ]
        with patch('sys.argv', args):
            explainaboard.explainaboard_main.main()

    def test_codegen_datalab(self):
        args = [
            'explainaboard.explainaboard_main',
            "--task",
            "machine-translation",
            "--dataset",
            "conala",
            "--output_file_type",
            "json",
            "--system_outputs",
            f"{top_path}/data/system_outputs/conala/conala-baseline-output.json",
            "--report_json",
            "report.json",
        ]
        with patch('sys.argv', args):
            explainaboard.explainaboard_main.main()

    def test_lm_custom(self):
        args = [
            'explainaboard.explainaboard_main',
            '--task',
            'language-modeling',
            '--custom_dataset_paths',
            f'{top_path}/data/system_outputs/wikitext/wikitext-dataset.txt',
            '--system_outputs',
            f'{top_path}/data/system_outputs/wikitext/wikitext-sys1-output.txt',
            '--report_json',
            '/dev/null',
        ]
        with patch('sys.argv', args):
            explainaboard.explainaboard_main.main()

    def test_ner_datalab(self):
        args = [
            'explainaboard.explainaboard_main',
            '--task',
            'named-entity-recognition',
            '--dataset',
            'conll2003',
            '--sub_dataset',
            'ner',
            '--system_outputs',
            f'{top_path}/data/system_outputs/conll2003/conll2003-elmo-output.conll',
            '--report_json',
            '/dev/null',
        ]
        with patch('sys.argv', args):
            explainaboard.explainaboard_main.main()

    def test_ner_custom(self):
        args = [
            'explainaboard.explainaboard_main',
            '--task',
            'named-entity-recognition',
            '--custom_dataset_paths',
            f'{top_path}/data/system_outputs/conll2003/conll2003-dataset.conll',
            '--system_outputs',
            f'{top_path}/data/system_outputs/conll2003/conll2003-elmo-output.conll',
            '--report_json',
            '/dev/null',
        ]
        with patch('sys.argv', args):
            explainaboard.explainaboard_main.main()

    def test_multichoiceqa_datalab(self):
        args = [
            'explainaboard.explainaboard_main',
            '--task',
            'qa-multiple-choice',
            '--dataset',
            'fig_qa',
            '--split',
            'validation',
            '--system_outputs',
            f'{top_path}/data/system_outputs/fig_qa/fig_qa-gptneo-output.json',  # noqa
            '--report_json',
            '/dev/null',
        ]
        with patch('sys.argv', args):
            explainaboard.explainaboard_main.main()

    def test_multichoiceqa_custom(self):
        args = [
            'explainaboard.explainaboard_main',
            '--task',
            'qa-multiple-choice',
            '--custom_dataset_paths',
            f'{top_path}/data/system_outputs/fig_qa/fig_qa-dataset.json',
            '--system_outputs',
            f'{top_path}/data/system_outputs/fig_qa/fig_qa-gptneo-output.json',  # noqa
            '--report_json',
            '/dev/null',
        ]
        with patch('sys.argv', args):
            explainaboard.explainaboard_main.main()

    def test_extractiveqa_custom(self):
        args = [
            'explainaboard.explainaboard_main',
            '--task',
            'qa-extractive',
            '--custom_dataset_paths',
            f'{top_path}/data/system_outputs/squad/squad_mini-dataset.json',
            '--system_outputs',
            f'{top_path}/data/system_outputs/squad/squad_mini-example-output.json',
            '--report_json',
            '/dev/null',
        ]
        with patch('sys.argv', args):
            explainaboard.explainaboard_main.main()

    @unittest.skip(
        reason="to be fixed in future PR: "
        "https://github.com/neulab/ExplainaBoard/issues/247"
    )
    def test_kglinktail_datalab(self):
        args = [
            'explainaboard.explainaboard_main',
            '--task',
            'kg-link-tail-prediction',
            '--dataset',
            'fb15k_237',
            '--system_outputs',
            f'{top_path}/data/system_outputs/fb15k-237/test-kg-prediction-no-user-defined.json',  # noqa
            '--report_json',
            '/dev/null',
        ]
        with patch('sys.argv', args):
            explainaboard.explainaboard_main.main()

    def test_kglinktail_custom(self):
        args = [
            'explainaboard.explainaboard_main',
            '--task',
            'kg-link-tail-prediction',
            '--custom_dataset_paths',
            f'{top_path}/data/system_outputs/fb15k-237/test-kg-prediction-no-user-defined.json',  # noqa
            '--system_outputs',
            f'{top_path}/data/system_outputs/fb15k-237/test-kg-prediction-no-user-defined.json',  # noqa
            '--report_json',
            '/dev/null',
        ]
        with patch('sys.argv', args):
            explainaboard.explainaboard_main.main()

    def test_absa_custom(self):
        args = [
            'explainaboard.explainaboard_main',
            '--task',
            'aspect-based-sentiment-classification',
            '--custom_dataset_paths',
            f'{top_path}/data/system_outputs/absa/absa-dataset.tsv',
            '--system_outputs',
            f'{top_path}/data/system_outputs/absa/absa-example-output.txt',
            '--report_json',
            '/dev/null',
        ]
        with patch('sys.argv', args):
            explainaboard.explainaboard_main.main()