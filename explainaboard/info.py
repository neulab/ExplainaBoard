"""Classes to hold information from analyses, etc."""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
import json
import os
import sys
from typing import Any, Callable, Optional

from explainaboard import config
from explainaboard.analysis.analyses import Analysis, AnalysisLevel
from explainaboard.analysis.case import AnalysisCase
from explainaboard.analysis.result import Result
from explainaboard.metrics.metric import MetricStats
from explainaboard.serialization.legacy import general_to_dict
from explainaboard.utils.logging import get_logger
from explainaboard.utils.tokenizer import get_tokenizer_serializer, Tokenizer

logger = get_logger(__name__)


@dataclass
class Table:
    """A table of information.

    Attributes:
        table: A dictionary holding the table.
    """

    table: Optional[dict] = None


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


@dataclass
class SysOutputInfo:
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

    # set in the system_output scripts
    task_name: str
    system_name: Optional[str] = None
    dataset_name: Optional[str] = None
    sub_dataset_name: Optional[str] = None
    dataset_split: Optional[str] = None
    source_language: str | None = None
    target_language: str | None = None
    reload_stat: bool = True
    confidence_alpha: float = 0.05
    system_details: Optional[dict] = None
    source_tokenizer: Optional[Tokenizer] = None
    target_tokenizer: Optional[Tokenizer] = None
    analysis_levels: Optional[list[AnalysisLevel]] = None
    analyses: Optional[list[Analysis]] = None

    # set later
    results: Result = field(default_factory=lambda: Result())

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

    @classmethod
    def from_directory(cls, sys_output_info_dir: str) -> "SysOutputInfo":
        """Create SysOutputInfo from the JSON file in `sys_output_info_dir`.

        Args:
            sys_output_info_dir (`str`): The directory containing the metadata file.
                This should be the root directory of a specific dataset version.
        """
        logger.info("Loading Dataset info from %s", sys_output_info_dir)
        if not sys_output_info_dir:
            raise ValueError(
                "Calling DatasetInfo.from_directory() with undefined dataset_info_dir."
            )

        with open(
            os.path.join(sys_output_info_dir, config.SYS_OUTPUT_INFO_FILENAME),
            "r",
            encoding="utf-8",
        ) as f:
            data_dict = json.load(f)
        return cls.from_dict(data_dict)

    @classmethod
    def dict_conv(cls, k: str, v: Any) -> Any:
        """Deserialization utility function.

        A deserialization utility function that takes in a key corresponding to a
        parameter name, and dictionary corresponding to a serialized version of that
        parameter's value, then return the deserialized version of the value.

        Args:
            k: the parameter name
            v: the parameter's value

        Returns:
            The modified parameter value
        """
        if k == 'results':
            return Result.from_dict(v)
        elif k.endswith('tokenizer'):
            return get_tokenizer_serializer().deserialize(v)
        elif k == 'analysis_levels':
            return [AnalysisLevel.from_dict(v1) for v1 in v]
        elif k == 'analyses':
            return [Analysis.from_dict(v1) for v1 in v]
        else:
            return v

    @classmethod
    def from_dict(cls, data_dict: dict) -> SysOutputInfo:
        """Deserialization function."""
        field_names = set(f.name for f in dataclasses.fields(cls))
        return cls(
            **{k: cls.dict_conv(k, v) for k, v in data_dict.items() if k in field_names}
        )

    def update(self, other_sys_output_info: SysOutputInfo, ignore_none=True):
        """Update with another SysOutputInfo."""
        self_dict = self.__dict__
        self_dict.update(
            **{
                k: copy.deepcopy(v)
                for k, v in other_sys_output_info.__dict__.items()
                if (v is not None or not ignore_none)
            }
        )

    def copy(self) -> SysOutputInfo:
        """Create a new copy of the SysOutputInfo."""
        return self.__class__(**{k: copy.deepcopy(v) for k, v in self.__dict__.items()})


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
    metric_stats: list[list[MetricStats]]
