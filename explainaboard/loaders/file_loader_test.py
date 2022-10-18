"""Tests for explainaboard.loaders.file_loader."""

from __future__ import annotations

from unittest import TestCase

from explainaboard import Source
from explainaboard.loaders.file_loader import (
    FileLoaderField,
    TextFileLoader,
    TSVFileLoader,
)


class FileLoaderTest(TestCase):
    def test_tsv_validation(self):
        self.assertRaises(
            ValueError,
            lambda: TSVFileLoader(
                [FileLoaderField("0", "field0", str)], use_idx_as_id=True
            ),
        )

    def test_text_file_loader_str(self):
        content = "line1\nline2"
        loader = TextFileLoader(target_name="output", dtype=str)
        data = loader.load(content, Source.in_memory)
        self.assertEqual(
            [
                {"id": "0", "output": "line1"},
                {"id": "1", "output": "line2"},
            ],
            data.samples,
        )

    def test_text_file_loader_int(self):
        content = "1\n2\n"
        loader = TextFileLoader(target_name="prediction", dtype=int)
        data = loader.load(content, Source.in_memory)
        self.assertEqual(
            [
                {"id": "0", "prediction": 1},
                {"id": "1", "prediction": 2},
            ],
            data.samples,
        )

    def test_text_file_loader_validate(self):
        loader = TextFileLoader(target_name="prediction", dtype=int)
        self.assertRaises(
            ValueError,
            lambda: loader.add_fields([FileLoaderField("test", "test", str)]),
        )
