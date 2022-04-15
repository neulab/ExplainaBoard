import unittest
from unittest import TestCase
from unittest.mock import patch

from explainaboard.explainaboard_main import main
from explainaboard.tests.utils import OPTIONAL_TEST_SUITES


class TestCLI(TestCase):
    """TODO: these tests only tests if they run. After the main script
    has been refactored, we can make these tests more useful"""

    def test_textclass_datalab(self):
        args = [
            'explainaboard.explainaboard_main',
            '--task',
            'text-classification',
            '--system_outputs',
            './data/system_outputs/sst2/sst2-lstm-output.txt',
            '--dataset',
            'sst2',
        ]
        with patch('sys.argv', args):
            main()

    def test_textclass_custom(self):
        args = [
            'explainaboard.explainaboard_main',
            '--task',
            'text-classification',
            '--system_outputs',
            './data/system_outputs/sst2/sst2-lstm-output.txt',
            '--custom_dataset_paths',
            './data/system_outputs/sst2/sst2-dataset.tsv',
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
            './data/system_outputs/snli/snli-bert-output.txt',
            '--dataset',
            'snli',
        ]
        with patch('sys.argv', args):
            main()

    def test_textpair_custom(self):
        args = [
            'explainaboard.explainaboard_main',
            '--task',
            'text-pair-classification',
            '--system_outputs',
            './data/system_outputs/snli/snli-bert-output.txt',
            '--custom_dataset_paths',
            './data/system_outputs/snli/snli-dataset.tsv',
        ]
        with patch('sys.argv', args):
            main()

    def test_summ_custom(self):
        args = [
            'explainaboard.explainaboard_main',
            '--task',
            'summarization',
            '--custom_dataset_paths',
            './data/system_outputs/cnndm/cnndm_mini-dataset.tsv',
            '--system_outputs',
            './data/system_outputs/cnndm/cnndm_mini-bart-output.txt',
            '--metrics',
            'rouge2',
            'bart_score_en_ref',
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
            './data/system_outputs/conll2003/conll2003-elmo-output.conll',
        ]
        with patch('sys.argv', args):
            main()

    def test_ner_custom(self):
        args = [
            'explainaboard.explainaboard_main',
            '--task',
            'named-entity-recognition',
            '--custom_dataset_paths',
            './data/system_outputs/conll2003/conll2003-dataset.conll',
            '--system_outputs',
            './data/system_outputs/conll2003/conll2003-elmo-output.conll',
        ]
        with patch('sys.argv', args):
            main()
