"""Classes to hold information from analyses, etc."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
import json
import os
import sys
from typing import Any, cast, ClassVar, final, Optional, TextIO, TypeVar

from explainaboard import config
from explainaboard.analysis.analyses import Analysis, AnalysisLevel
from explainaboard.analysis.case import AnalysisCase
from explainaboard.analysis.result import Result
from explainaboard.metrics.metric import MetricStats
from explainaboard.serialization import common_registry
from explainaboard.serialization.serializers import PrimitiveSerializer
from explainaboard.serialization.types import Serializable, SerializableData
from explainaboard.utils.logging import get_logger
from explainaboard.utils.tokenizer import Tokenizer
from explainaboard.utils.typing_utils import narrow, unwrap_or

logger = get_logger(__name__)

T = TypeVar("T")


# TODO(odashi): This function may be generally useful. Move it to the serialization
# submodule.
def _get_value(data: dict[str, SerializableData], cls: type[T], key: str) -> T | None:
    """Helper function to obtain a typed value or None from a serialized data.

    Args:
        data: Serialized data.
        cls: Data type to obtain.
        key: Data key to obtain.

    Returns:
        `data[key]` if it has type of `cls`, `None` if `data[key]` does not exist.

    Raises:
        TypeError: Thrown by inner `narrow()`: `data[key]` has an incompatible type.
    """
    value = data.get(key)
    return narrow(cls, value) if value is not None else None


@dataclass
class PaperInfo:
    """Information about a paper.

    Attributes:
        year: the year the paper was published
        venue: where the paper was published
        title: the title of the paper
        author: the author(s) of the paper
        url: the url where the paper can be found
        bib: bibliography information
    """

    year: Optional[str] = None
    venue: Optional[str] = None
    title: Optional[str] = None
    author: Optional[str] = None
    url: Optional[str] = None
    bib: Optional[str] = None


@common_registry.register("SysOutputInfo")
@final
@dataclass
class SysOutputInfo(Serializable):
    """Information about a system output and its analysis settings.

    Attributes:
        task_name (str): the name of the task.
        system_name (str): the name of the system .
        dataset_name (str): the dataset used of the system.
        sub_dataset_name (str): the name of the sub-dataset.
        dataset_split (str): the name of the split.
        source_language (str): the language of the input
        target_language (str): the language of the output
        system_details (dict): a dictionary of system details
        source_tokenizer (Tokenizer): the tokenizer for source sentences
        target_tokenizer (Tokenizer): the tokenizer for target sentences
        analysis_levels: the levels of analysis to perform
    """

    DEFAULT_CONFIDENCE_ALPHA: ClassVar[float] = 0.05

    task_name: str | None = None
    system_name: str | None = None
    dataset_name: str | None = None
    sub_dataset_name: str | None = None
    dataset_split: str | None = None
    source_language: str | None = None
    target_language: str | None = None
    # NOTE(odashi): confidence_alpha == None has a meaning beyond "unset": it prevents
    # calculating confidence intervals.
    confidence_alpha: float | None = DEFAULT_CONFIDENCE_ALPHA
    system_details: dict[str, SerializableData] = field(default_factory=dict)
    source_tokenizer: Tokenizer | None = None
    target_tokenizer: Tokenizer | None = None
    analysis_levels: list[AnalysisLevel] = field(default_factory=list)
    analyses: list[Analysis] = field(default_factory=list)

    # set later
    results: Result = field(default_factory=lambda: Result(overall={}, analyses=[]))

    # TODO(odashi): This function does many out-of-scope work. It should be enough to
    # provide a functionality to dump the serialized data into a dict, and let users
    # save the dumped data under their responsibility.
    def write_to_directory(
        self,
        dataset_info_dir: str,
        file_name: str | None = None,
        overwrite: bool = False,
    ) -> None:
        """Write `SysOutputInfo` as JSON to `dataset_info_dir`.

        This function is not thread-safe. Modification of the target directory/files by
        other processes/threads may cause unintended behavior.

        Args:
            dataset_info_dir: Directory path to store the JSON file. If the directory
                does not exist, this function craetes it recursively.
            file_name: JSON file name under `dataset_info_dir`. If None, default file
                name is used.
            overwrite: If True, this function overwrites the existing file. If Fasle, it
                raises an Exception if the file already exists.

        Raises:
            RuntimeError: File already exists.
        """
        file_name = (
            file_name if file_name is not None else config.SYS_OUTPUT_INFO_FILENAME
        )
        file_path = os.path.join(dataset_info_dir, file_name)

        # Checks the directory.
        if os.path.exists(dataset_info_dir):
            if not os.path.isdir(dataset_info_dir):
                raise RuntimeError(f"Not a directory: {dataset_info_dir}")
        else:
            os.makedirs(dataset_info_dir)

        # Checks the file.
        if os.path.exists(file_path):
            if not os.path.isfile(file_path):
                raise RuntimeError(f"Not a file: {file_path}")
            if not overwrite:
                raise RuntimeError(
                    f"Attempted to overwrite the existing file: {file_path}"
                )

        with open(file_path, "w") as f:
            self.print_as_json(file=f)

    def print_as_json(self, file: TextIO | None = None) -> None:
        """Print as json to the specified file.

        Args:
            file: The file stream to print to, or None for stdout.
        """
        json.dump(
            PrimitiveSerializer().serialize(self),
            file if file is not None else sys.stdout,
            indent=2,
        )

    def serialize(self) -> dict[str, SerializableData]:
        """Implements Serializable.serialize."""
        return {
            "task_name": self.task_name,
            "system_name": self.system_name,
            "dataset_name": self.dataset_name,
            "sub_dataset_name": self.sub_dataset_name,
            "dataset_split": self.dataset_split,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "confidence_alpha": self.confidence_alpha,
            "system_details": self.system_details,
            "source_tokenizer": self.source_tokenizer,
            "target_tokenizer": self.target_tokenizer,
            "analysis_levels": self.analysis_levels,
            "analyses": self.analyses,
            "results": self.results,
        }

    @classmethod
    def deserialize(cls, data: dict[str, SerializableData]) -> Serializable:
        """Implements Serializable.deserialize."""
        # TODO(odashi): Remove type:ignore if mypy/4717 was fixed.

        if "confidence_alpha" in data:
            confidence_alpha = _get_value(data, float, "confidence_alpha")
        else:
            confidence_alpha = cls.DEFAULT_CONFIDENCE_ALPHA

        system_details = {
            narrow(str, k): cast(SerializableData, v)
            for k, v in unwrap_or(_get_value(data, dict, "system_details"), {}).items()
        }
        analysis_levels = [
            narrow(AnalysisLevel, x)
            for x in unwrap_or(_get_value(data, list, "analysis_levels"), [])
        ]
        analyses = [
            narrow(Analysis, x)  # type: ignore
            for x in unwrap_or(_get_value(data, list, "analyses"), [])
        ]

        return cls(
            task_name=_get_value(data, str, "task_name"),
            system_name=_get_value(data, str, "system_name"),
            dataset_name=_get_value(data, str, "dataset_name"),
            sub_dataset_name=_get_value(data, str, "sub_dataset_name"),
            dataset_split=_get_value(data, str, "dataset_split"),
            source_language=_get_value(data, str, "source_language"),
            target_language=_get_value(data, str, "target_language"),
            confidence_alpha=confidence_alpha,
            system_details=system_details,
            source_tokenizer=_get_value(
                data, Tokenizer, "source_tokenizer"  # type: ignore
            ),
            target_tokenizer=_get_value(
                data, Tokenizer, "target_tokenizer"  # type: ignore
            ),
            analysis_levels=analysis_levels,
            analyses=analyses,
            results=unwrap_or(
                _get_value(data, Result, "results"), Result(overall={}, analyses=[])
            ),
        )

    # TODO(odashi): This function is hacky and shouldn't be used.
    # Remove this function after introducing the struct of system metadata.
    # See also: https://github.com/neulab/ExplainaBoard/issues/575
    @classmethod
    def from_any_dict(cls, data: dict[str, Any]) -> SysOutputInfo:
        """Generates SysOutputInfo from a dict.

        Args:
            data: Data, which may contain some information about SysOutputInfo.

        Returns:
            Generated SysOutputInfo.
        """
        keys = set(x.name for x in dataclasses.fields(cls))

        serialized_sysout = {
            cast(str, k): cast(SerializableData, v)
            for k, v in data.items()
            if k in keys
        }
        serialized_sysout["cls_name"] = "SysOutputInfo"

        return narrow(
            SysOutputInfo, PrimitiveSerializer().deserialize(serialized_sysout)
        )


@dataclass
class OverallStatistics:
    """Overall statistics calculated by the processor.

    Attributes:
        sys_info: The system info
        analysis_cases: The extracted analysis cases
        metric_stats: The statistics needed to calculate each metric
    """

    sys_info: SysOutputInfo
    analysis_cases: list[list[AnalysisCase]]
    metric_stats: list[dict[str, MetricStats]]
