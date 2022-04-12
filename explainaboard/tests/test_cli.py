from unittest import TestCase
from unittest.mock import patch

from explainaboard_main import main


class TestCLI(TestCase):
    """TODO: these tests only tests if they run. After the main script
    has been refactored, we can make these tests more useful"""

    def test_processing_datalab_dataset(self):
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

    def test_processing_custom_dataset(self):
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
