"""Tests for io_utils."""

import os
import sys
import tempfile
import unittest

from explainaboard.utils import io_utils


class TestIOUtils(unittest.TestCase):
    def test_text_writer_file(self):
        with tempfile.TemporaryDirectory as dirname:
            with io_utils.text_writer(os.path.join(dirname, "test.txt")) as fp:
                fp.write("foobar")
            self.assertEqual(open(dirname, "test.txt").read(), "foobar")

    def test_text_writer_stdout(self):
        with io_utils.text_writer() as fp:
            self.assertIs(fp, sys.stdout)
