from unittest import TestCase

from explainaboard.loaders.file_loader import FileLoaderField, TSVFileLoader


class FileLoaderTests(TestCase):
    def test_tsv_validation(self):
        self.assertRaises(
            ValueError,
            lambda: TSVFileLoader(
                [FileLoaderField("0", "field0", str)], use_idx_as_id=True
            ),
        )
