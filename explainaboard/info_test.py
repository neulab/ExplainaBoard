"""Tests for explainaboard.info."""

from __future__ import annotations

import copy
import pathlib
import tempfile
import unittest

from explainaboard.analysis.analyses import Analysis, AnalysisLevel, BucketAnalysis
from explainaboard.analysis.result import Result
from explainaboard.config import SYS_OUTPUT_INFO_FILENAME
from explainaboard.info import SysOutputInfo
from explainaboard.serialization.serializers import PrimitiveSerializer
from explainaboard.utils.tokenizer import SacreBleuTokenizer, SingleSpaceTokenizer
from explainaboard.utils.typing_utils import narrow


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

    def test_serialization(self) -> None:
        tokenizer1 = SingleSpaceTokenizer()
        tokenizer2 = SacreBleuTokenizer()
        levels = [AnalysisLevel("level", {}, {})]
        analyses: list[Analysis] = [BucketAnalysis("description", "level", "feature")]
        results = Result(overall={}, analyses=[])
        sysout_base = SysOutputInfo(
            task_name="foo",
            system_name="bar",
            dataset_name="baz",
            sub_dataset_name="qux",
            dataset_split="quux",
            source_language="en",
            target_language="zh",
            confidence_alpha=None,
            system_details={"detail": 123},
            source_tokenizer=tokenizer1,
            target_tokenizer=tokenizer2,
            analysis_levels=levels,
            analyses=analyses,
            results=results,
        )

        serializer = PrimitiveSerializer()

        tokenizer1_serialized = serializer.serialize(tokenizer1)
        tokenizer2_serialized = serializer.serialize(tokenizer2)
        levels_serialized = serializer.serialize(levels)
        analyses_serialized = serializer.serialize(analyses)
        results_serialized = serializer.serialize(results)
        sysout_serialized_base = {
            "cls_name": "SysOutputInfo",
            "task_name": "foo",
            "system_name": "bar",
            "dataset_name": "baz",
            "sub_dataset_name": "qux",
            "dataset_split": "quux",
            "source_language": "en",
            "target_language": "zh",
            "system_details": {"detail": 123},
            "source_tokenizer": tokenizer1_serialized,
            "target_tokenizer": tokenizer2_serialized,
            "analysis_levels": levels_serialized,
            "analyses": analyses_serialized,
            "results": results_serialized,
        }

        # Test serialization
        for alpha in [None, 0.5]:
            sysout_serialized = dict(sysout_serialized_base)
            sysout_serialized["confidence_alpha"] = alpha
            sysout = copy.copy(sysout_base)
            sysout.confidence_alpha = alpha
            self.assertEqual(serializer.serialize(sysout), sysout_serialized)

        test_cases: list[tuple[bool, float | None, float | None]] = [
            (False, None, SysOutputInfo.DEFAULT_CONFIDENCE_ALPHA),
            (True, None, None),
            (True, 0.5, 0.5),
        ]

        # Test deserialization
        for set_alpha, given_alpha, restored_alpha in test_cases:
            sysout_serialized = dict(sysout_serialized_base)
            if set_alpha:
                sysout_serialized["confidence_alpha"] = given_alpha
            sysout = copy.copy(sysout_base)
            sysout.confidence_alpha = restored_alpha

            # SysOutputInfo can't be compared directly.
            deserialized = narrow(
                SysOutputInfo, serializer.deserialize(sysout_serialized)
            )
            self.assertEqual(deserialized.task_name, sysout.task_name)
            self.assertEqual(deserialized.system_name, sysout.system_name)
            self.assertEqual(deserialized.dataset_name, sysout.dataset_name)
            self.assertEqual(deserialized.sub_dataset_name, sysout.sub_dataset_name)
            self.assertEqual(deserialized.dataset_split, sysout.dataset_split)
            self.assertEqual(deserialized.source_language, sysout.source_language)
            self.assertEqual(deserialized.target_language, sysout.target_language)
            self.assertEqual(deserialized.confidence_alpha, sysout.confidence_alpha)
            self.assertEqual(deserialized.system_details, sysout.system_details)
            self.assertIsInstance(deserialized.source_tokenizer, SingleSpaceTokenizer)
            self.assertIsInstance(deserialized.target_tokenizer, SacreBleuTokenizer)
            self.assertEqual(deserialized.analysis_levels, sysout.analysis_levels)
            self.assertEqual(deserialized.analyses, sysout.analyses)
            self.assertEqual(deserialized.results, sysout.results)

    def test_from_any_dict(self) -> None:
        data = {
            "task_name": "foo",
            "system_name": "bar",
            "unknown_field": "baz",  # should be ignored
        }
        deserialized = SysOutputInfo.from_any_dict(data)
        self.assertEqual(deserialized.task_name, "foo")
        self.assertEqual(deserialized.system_name, "bar")
        self.assertIsNone(deserialized.dataset_name)
        self.assertIsNone(deserialized.sub_dataset_name)
        self.assertIsNone(deserialized.dataset_split)
        self.assertIsNone(deserialized.source_language)
        self.assertIsNone(deserialized.target_language)
        self.assertEqual(
            deserialized.confidence_alpha, SysOutputInfo.DEFAULT_CONFIDENCE_ALPHA
        )
        self.assertEqual(deserialized.system_details, {})
        self.assertIsNone(deserialized.source_tokenizer)
        self.assertIsNone(deserialized.target_tokenizer)
        self.assertEqual(deserialized.analysis_levels, [])
        self.assertEqual(deserialized.analyses, [])
        self.assertEqual(deserialized.results, Result(overall={}, analyses=[]))
