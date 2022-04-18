import os
import unittest
from unittest import TestCase
from unittest.mock import patch

import requests

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
            './data/system_outputs/sst2/sst2-lstm-output.txt',
            '--custom_dataset_paths',
            './data/system_outputs/sst2/sst2-dataset.tsv',
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
            './data/system_outputs/snli/snli-roberta-output.txt',
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
            './data/system_outputs/snli/snli-roberta-output.txt',
            '--custom_dataset_paths',
            './data/system_outputs/snli/snli-dataset.tsv',
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
            './data/system_outputs/cnndm/cnndm_mini-dataset.tsv',
            '--system_outputs',
            './data/system_outputs/cnndm/cnndm_mini-bart-output.txt',
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
        fname = 'cnndm-bart-output.txt'
        local_file = f'./data/system_outputs/cnndm/{fname}'
        if not os.path.exists(local_file):
            url = 'http://www.phontron.com/download/' + fname
            r = requests.get(url)
            open(local_file, 'wb').write(r.content)
        args = [
            'explainaboard.explainaboard_main',
            '--task',
            'summarization',
            '--dataset',
            'cnn_dailymail',
            '--system_outputs',
            './data/system_outputs/cnndm/cnndm-bart-output.txt',
            '--metrics',
            'rouge2',
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
            './data/system_outputs/conll2003/conll2003-elmo-output.conll',
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
            './data/system_outputs/conll2003/conll2003-dataset.conll',
            '--system_outputs',
            './data/system_outputs/conll2003/conll2003-elmo-output.conll',
            '--report_json',
            '/dev/null',
        ]
        with patch('sys.argv', args):
            main()

    @unittest.skip(reason='metaphor QA dataset is temporarily unavailable')
    def test_multichoiceqa_datalab(self):
        args = [
            'explainaboard.explainaboard_main',
            '--task',
            'qa-multiple-choice',
            '--dataset',
            'metaphor_qa',
            '--system_outputs',
            './data/system_outputs/metaphor_qa/metaphor_qa-gptneo-output.json',
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
            './data/system_outputs/metaphor_qa/metaphor_qa-dataset.json',
            '--system_outputs',
            './data/system_outputs/metaphor_qa/metaphor_qa-gptneo-output.json',
            '--report_json',
            '/dev/null',
        ]
        with patch('sys.argv', args):
            main()

    def test_extractiveqa_custom(self):
        args = [
            'explainaboard.explainaboard_main',
            '--task',
            'question-answering-extractive',
            '--custom_dataset_paths',
            './data/system_outputs/squad/squad_mini-dataset.json',
            '--system_outputs',
            './data/system_outputs/squad/squad_mini-example-output.json',
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
            './data/system_outputs/fb15k-237/test-kg-prediction-no-user-defined-new.json',  # noqa
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
            './data/system_outputs/fb15k-237/test-kg-prediction-no-user-defined-new.json',  # noqa
            '--system_outputs',
            './data/system_outputs/fb15k-237/test-kg-prediction-no-user-defined-new.json',  # noqa
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
            './data/system_outputs/absa/absa-dataset.tsv',
            '--system_outputs',
            './data/system_outputs/absa/absa-example-output.txt',
            '--report_json',
            '/dev/null',
        ]
        with patch('sys.argv', args):
            main()
