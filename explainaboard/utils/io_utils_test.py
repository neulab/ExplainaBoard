"""Tests for io_utils."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

from explainaboard.utils import io_utils


class IOUtilsTest(unittest.TestCase):
    def test_text_writer_file(self):
        with tempfile.TemporaryDirectory() as dirname:
            filename = os.path.join(dirname, "test.txt")
            with io_utils.text_writer(filename) as fp:
                fp.write("foobar")
            with open(filename) as fp:
                self.assertEqual(fp.read(), "foobar")

    def test_text_writer_stdout(self):
        with io_utils.text_writer() as fp:
            self.assertIs(fp, sys.stdout)
