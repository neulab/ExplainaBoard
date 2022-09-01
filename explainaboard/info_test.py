"""Tests for explainaboard.info."""

import pathlib
import tempfile
import unittest

from explainaboard.config import SYS_OUTPUT_INFO_FILENAME
from explainaboard.info import SysOutputInfo


class SysOutputInfoTest(unittest.TestCase):
    def test_write_to_directory(self) -> None:
        info = SysOutputInfo(task_name="test")

        def assert_content(path: pathlib.Path) -> None:
            """Helper to check if the file is written correctly."""
            self.assertTrue(path.exists())
            with path.open() as f:
                self.assertEqual(f.readline(), "{\n")  # Checks only the first line.

        with tempfile.TemporaryDirectory() as tmpdir:
            td = pathlib.Path(tmpdir)

            # w/o directory
            dir1 = td / "dir1"
            dir2 = td / "dir2"
            info.write_to_directory(str(dir1))
            info.write_to_directory(str(dir2), "my.json")
            assert_content(dir1 / SYS_OUTPUT_INFO_FILENAME)
            assert_content(dir2 / "my.json")

            # w/ directory, w/o file
            dir3 = td / "dir3"
            dir3.mkdir()
            info.write_to_directory(str(dir3))
            info.write_to_directory(str(dir3), "my.json")
            assert_content(dir3 / SYS_OUTPUT_INFO_FILENAME)
            assert_content(dir3 / "my.json")

            # w/ directory, w/ file
            dir4 = td / "dir4"
            dir4.mkdir()
            (dir4 / SYS_OUTPUT_INFO_FILENAME).touch()
            (dir4 / "my.json").touch()
            with self.assertRaisesRegex(RuntimeError, r"^Attempted to overwrite"):
                info.write_to_directory(str(dir4))
            with self.assertRaisesRegex(RuntimeError, r"^Attempted to overwrite"):
                info.write_to_directory(str(dir4), "my.json")
            info.write_to_directory(str(dir4), overwrite=True)
            info.write_to_directory(str(dir4), "my.json", overwrite=True)
            assert_content(dir4 / SYS_OUTPUT_INFO_FILENAME)
            assert_content(dir4 / "my.json")

            # Overwrite directory with file
            dir5 = td / "dir5"
            dir5.mkdir()
            (dir5 / SYS_OUTPUT_INFO_FILENAME).mkdir()
            (dir5 / "my.json").mkdir()
            with self.assertRaisesRegex(RuntimeError, r"^Not a file"):
                info.write_to_directory(str(dir5))
            with self.assertRaisesRegex(RuntimeError, r"^Not a file"):
                info.write_to_directory(str(dir5), "my.json")
            with self.assertRaisesRegex(RuntimeError, r"^Not a file"):
                info.write_to_directory(str(dir5), overwrite=True)
            with self.assertRaisesRegex(RuntimeError, r"^Not a file"):
                info.write_to_directory(str(dir5), "my.json", overwrite=True)

            # Overwrite file with directory
            dir6 = td / "dir6"
            dir6.mkdir()
            (dir6 / "mydir").touch()
            with self.assertRaisesRegex(RuntimeError, r"^Not a directory"):
                info.write_to_directory(str(dir6 / "mydir"))
            with self.assertRaisesRegex(RuntimeError, r"^Not a directory"):
                info.write_to_directory(str(dir6 / "mydir"), "my.json")
            with self.assertRaisesRegex(RuntimeError, r"^Not a directory"):
                info.write_to_directory(str(dir6 / "mydir"), overwrite=True)
            with self.assertRaisesRegex(RuntimeError, r"^Not a directory"):
                info.write_to_directory(str(dir6 / "mydir"), "my.json", overwrite=True)
