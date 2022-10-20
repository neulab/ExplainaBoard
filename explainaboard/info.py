"""Classes to hold information from analyses, etc."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
import json
import os
import sys
from typing import Callable, ClassVar, final, Optional, TypeVar

from explainaboard import config
from explainaboard.analysis.analyses import Analysis, AnalysisLevel
from explainaboard.analysis.case import AnalysisCase
from explainaboard.analysis.result import Result
from explainaboard.metrics.metric import MetricStats
from explainaboard.serialization import common_registry
from explainaboard.serialization.legacy import general_to_dict
from explainaboard.serialization.types import Serializable, SerializableData
from explainaboard.utils.logging import get_logger
from explainaboard.utils.tokenizer import Tokenizer
from explainaboard.utils.typing_utils import narrow, unwrap

logger = get_logger(__name__)

T = TypeVar("T")


# TODO(odashi): This function may be generally useful. Move it to the serialization
# submodule.
def _get_value_or(
    data: dict[str, SerializableData], cls: type[T], key: str, default: T | None = None
) -> T | None:
    """Helper function to obtain a typed value or None from a serialized data.

    Args:
        data: Serialized data.
        cls: Data type to obtain.
        key: Data key to obtain.
        default: Default value or None.

    Returns:
        `data[key]` if it has type of `cls`, `default` if `data[key]` does not exist.

    Raises:
        TypeError: Thrown by inner `narrow()`: `data[key]` has an incompatible type.
    """
    value = data.get(key)
    return narrow(cls, value) if value is not None else default


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
        reload_stat (bool): whether to reload the statistics or not
        system_details (dict): a dictionary of system details
        source_tokenizer (Tokenizer): the tokenizer for source sentences
        target_tokenizer (Tokenizer): the tokenizer for target sentences
        analysis_levels: the levels of analysis to perform
    """

    DEFAULT_RELOAD_STAT: ClassVar[bool] = True
    DEFAULT_CONFIDENCE_ALPHA: ClassVar[float] = 0.05

    # set in the system_output scripts
    task_name: str
    system_name: str | None = None
    dataset_name: str | None = None
    sub_dataset_name: str | None = None
    dataset_split: str | None = None
    source_language: str | None = None
    target_language: str | None = None
    reload_stat: bool = DEFAULT_RELOAD_STAT
    confidence_alpha: float = DEFAULT_CONFIDENCE_ALPHA
    system_details: dict[str, SerializableData] = field(default_factory=dict)
    source_tokenizer: Tokenizer | None = None
    target_tokenizer: Tokenizer | None = None
    analysis_levels: list[AnalysisLevel] = field(default_factory=list)
    analyses: list[Analysis] = field(default_factory=list)

    # set later
    results: Result = field(default_factory=lambda: Result(overall={}, analyses=[]))

    def to_dict(self) -> dict:
        """Serialization function."""
        ret_dict = {}
        for f in dataclasses.fields(self):
            obj = getattr(self, f.name)
            if obj is not None:
                ret_dict[f.name] = general_to_dict(obj)
        return ret_dict

    def replace_nonstring_keys(self, data):
        """Function to replace keys that are not strings for serialization to JSON."""
        if isinstance(data, list):
            for value in data:
                if isinstance(value, dict):
                    self.replace_nonstring_keys(value)
        else:
            replace_keys = []
            for key, value in data.items():
                if isinstance(value, Callable):
                    # TODO(gneubig): cannot serialize functions so info is lost
                    data[key] = None
                if not isinstance(key, str):
                    replace_keys.append(key)
                if isinstance(value, dict) or isinstance(value, list):
                    self.replace_nonstring_keys(value)
            for key in replace_keys:
                data[str(key)] = data[key]
                del data[key]

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

        with open(file_path, "wb") as f:
            self._dump_info(f)

    def print_as_json(self, file=None) -> None:
        """Print as json to the specified file.

        Args:
            file: The file stream to print to, or None for stdout.

        Raises:
            TypeError: If the data dict can not be written to.
        """
        if file is None:
            file = sys.stdout
        data_dict = self.to_dict()
        self.replace_nonstring_keys(data_dict)
        json.dump(data_dict, fp=file, indent=2)

    def _dump_info(self, file):
        """Convert SystemOutputInfo => JSON."""
        data_dict = self.to_dict()
        self.replace_nonstring_keys(data_dict)
        file.write(json.dumps(data_dict, indent=2).encode("utf-8"))

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
            "reload_stat": self.reload_stat,
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

        system_details = {
            narrow(str, k): narrow(SerializableData, v)  # type: ignore
            for k, v in unwrap(_get_value_or(data, dict, "system_details", {})).items()
        }
        analysis_levels = [
            narrow(AnalysisLevel, x)
            for x in unwrap(_get_value_or(data, list, "analysis_levels", []))
        ]
        analyses = [
            narrow(Analysis, x)  # type: ignore
            for x in unwrap(_get_value_or(data, list, "analyses", []))
        ]

        return cls(
            task_name=narrow(str, data["task_name"]),
            system_name=_get_value_or(data, str, "system_name"),
            dataset_name=_get_value_or(data, str, "dataset_name"),
            sub_dataset_name=_get_value_or(data, str, "sub_dataset_name"),
            dataset_split=_get_value_or(data, str, "dataset_split"),
            source_language=_get_value_or(data, str, "source_language"),
            target_language=_get_value_or(data, str, "target_language"),
            reload_stat=unwrap(
                _get_value_or(data, bool, "reload_stat", cls.DEFAULT_RELOAD_STAT)
            ),
            confidence_alpha=unwrap(
                _get_value_or(
                    data, float, "confidence_alpha", cls.DEFAULT_CONFIDENCE_ALPHA
                )
            ),
            system_details=system_details,
            source_tokenizer=_get_value_or(
                data, Tokenizer, "source_tokenizer"  # type: ignore
            ),
            target_tokenizer=_get_value_or(
                data, Tokenizer, "target_tokenizer"  # type: ignore
            ),
            analysis_levels=analysis_levels,
            analyses=analyses,
            results=unwrap(
                _get_value_or(data, Result, "results", Result(overall={}, analyses=[]))
            ),
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
