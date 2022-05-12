import unittest
from unittest import TestCase
from unittest.mock import patch

from explainaboard.explainaboard_main import main
from explainaboard.tests.utils import OPTIONAL_TEST_SUITES, top_path
from explainaboard.utils.cache_api import cache_online_file


class TestCLI(TestCase):
    """TODO: these tests only tests if they run. After the main script
    has been refactored, we can make these tests more useful"""

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
            main()

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
            main()

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
            main()

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
            main()

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
            main()

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
            main()

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
            main()

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
            main()

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
            main()

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
            main()

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
            main()

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
            main()

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
            main()

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
            main()

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
            main()

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
            main()

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
            main()

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
            main()

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
            main()
