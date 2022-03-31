from __future__ import annotations

import copy
import dataclasses
from dataclasses import asdict, dataclass, field
import json
import os
from typing import Any, Optional

from explainaboard import config
from explainaboard.feature import Features
from explainaboard.utils.logging import get_logger
from explainaboard.utils.py_utils import eprint

logger = get_logger(__name__)


@dataclass
class Table:
    table: Optional[dict] = None


@dataclass
class PaperInfo:
    """
    "year": "xx",
    "venue": "xx",
    "title": "xx",
    "author": "xx",
    "url": "xx",
    "bib": "xx"
    """

    year: Optional[str] = None
    venue: Optional[str] = None
    title: Optional[str] = None
    author: Optional[str] = None
    url: Optional[str] = None
    bib: Optional[str] = None


@dataclass
class Performance:
    metric_name: str
    value: float
    confidence_score_low: Optional[float] = None
    confidence_score_high: Optional[float] = None


@dataclass
class BucketPerformance:
    bucket_name: str
    n_samples: float
    bucket_samples: list[Any]
    performances: list[Performance] = field(default_factory=list)


@dataclass
class Result:
    overall: Any = None
    calibration: Optional[list[Performance]] = None
    fine_grained: Any = None


@dataclass
class SysOutputInfo:
    """Information about a system output

    Attributes:
        model_name (str): the name of the system .
        dataset_name (str): the dataset used of the system.
        language (str): the language of the dataset.
        code (str): the url of the code.
        download_link (str): the url of the system output.
        paper (Paper, optional): the published paper of the system.
        features (Features, optional): the features used to describe system output's
            column type.
        is_print_case (bool): Whether or not to print out cases
        is_print_confidence_interval (bool): Whether or not to print out confidence
            intervals
    """

    # set in the system_output scripts
    task_name: str
    model_name: Optional[str] = None
    dataset_name: Optional[str] = None
    sub_dataset_name: Optional[str] = None
    metric_names: Optional[list[str]] = None
    reload_stat: bool = True
    is_print_case: bool = True
    conf_value: float = 0.05
    # language : str = "English"

    # set later
    # code: str = None
    # download_link: str = None
    # paper_info: PaperInfo = PaperInfo()
    features: Optional[Features] = None
    results: Result = field(default_factory=lambda: Result())

    def to_dict(self) -> dict:
        return asdict(self)

    def replace_bad_keys(self, data):
        if isinstance(data, list):
            for value in data:
                if isinstance(value, dict):
                    self.replace_bad_keys(value)
        else:
            replace_keys = []
            for key, value in data.items():
                if not isinstance(key, str):
                    replace_keys.append(key)
                if isinstance(value, dict) or isinstance(value, list):
                    self.replace_bad_keys(value)
            for key in replace_keys:
                data[str(key)] = data[key]
                del data[key]

    def write_to_directory(self, dataset_info_dir):
        """Write `SysOutputInfo` as JSON to `dataset_info_dir`."""
        with open(
            os.path.join(dataset_info_dir, config.SYS_OUTPUT_INFO_FILENAME), "wb"
        ) as f:
            self._dump_info(f)

    def print_as_json(self):
        data_dict = self.to_dict()
        self.replace_bad_keys(data_dict)
        try:
            print(json.dumps(data_dict, indent=4))
        except TypeError as e:
            raise e

    def _dump_info(self, file):
        """SystemOutputInfo => JSON"""
        data_dict = self.to_dict()
        self.replace_bad_keys(data_dict)
        file.write(json.dumps(data_dict, indent=4).encode("utf-8"))

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
            sys_output_info_dict = json.load(f)
        return cls.from_dict(sys_output_info_dict)

    @classmethod
    def from_dict(cls, sys_output_info_dict: dict) -> "SysOutputInfo":
        field_names = set(f.name for f in dataclasses.fields(cls))
        return cls(
            **{k: v for k, v in sys_output_info_dict.items() if k in field_names}
        )

    def update(self, other_sys_output_info: "SysOutputInfo", ignore_none=True):
        self_dict = self.__dict__
        self_dict.update(
            **{
                k: copy.deepcopy(v)
                for k, v in other_sys_output_info.__dict__.items()
                if (v is not None or not ignore_none)
            }
        )

    def copy(self) -> "SysOutputInfo":
        return self.__class__(**{k: copy.deepcopy(v) for k, v in self.__dict__.items()})


@dataclass
class OverallStatistics:
    sys_info: SysOutputInfo
    metric_stats: Any = None
    active_features: Optional[list[str]] = None
    overall_results: Optional[dict[str, Performance]] = None


@dataclass
class FineGrainedStatistics:
    samples_over_bucket: dict
    performance_over_bucket: dict


def print_bucket_dict(dict_obj: dict[str, BucketPerformance], print_information: str):
    metric_names = [x.metric_name for x in next(iter(dict_obj.values())).performances]
    for i, metric_name in enumerate(metric_names):
        # print("-----------------------------------------------")
        eprint(f"the information of #{print_information}#")
        eprint(f"bucket_interval\t{metric_name}\t#samples")
        for k, v in dict_obj.items():
            if len(k) == 1:
                eprint(f"[{k[0]},]\t{v.performances[i].value}\t{v.n_samples}")
            else:
                eprint(f"[{k[0]},{k[1]}]\t{v.performances[i].value}\t{v.n_samples}")

        eprint("")
