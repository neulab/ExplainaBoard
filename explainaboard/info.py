from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
import json
import os
import sys
from typing import Any, Optional

from explainaboard import config
from explainaboard.feature import Features
from explainaboard.metric import MetricConfig, MetricStats
from explainaboard.utils.logging import get_logger
from explainaboard.utils.serialization import general_to_dict
from explainaboard.utils.tokenizer import Tokenizer

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

    @classmethod
    def from_dict(cls, data_dict: dict) -> Performance:
        field_names = set(f.name for f in dataclasses.fields(cls))
        return cls(**{k: v for k, v in data_dict.items() if k in field_names})


@dataclass
class BucketCase:
    """
    A class to represent cases to display to users for analysis.
    :param sample_id: The ID of a single sample
    """

    sample_id: int

    def __post_init__(self):
        if isinstance(self.sample_id, str):
            raise ValueError

    @classmethod
    def dict_conv(cls, k: str, v: dict):
        return v

    @classmethod
    def from_dict(cls, data_dict: dict) -> BucketCase:
        field_names = set(f.name for f in dataclasses.fields(cls))
        return cls(
            **{k: cls.dict_conv(k, v) for k, v in data_dict.items() if k in field_names}
        )


@dataclass
class BucketCaseSpan(BucketCase):
    """
    A bucket case that highlights a span in text.
    :param text: The text that should be highlighted
    :param token_span: The span of tokens to be highlighted
    :param char_span: The span of characters to be highlighted
    :param location: The name of the feature (e.g. "text", "source", "reference") over
      which this span is calculated
    """

    token_span: tuple[int, int]
    char_span: tuple[int, int]
    orig_str: str
    text: str

    def __post_init__(self):
        if isinstance(self.token_span, str) or isinstance(self.char_span, str):
            raise ValueError


@dataclass
class BucketCaseMultiSpan(BucketCase):
    """
    A bucket case that highlights multiple spans in text
    :param spans: The spans that are highlighted
    """

    spans: list[BucketCaseSpan]


@dataclass
class BucketCaseLabeledSpan(BucketCaseSpan):
    """
    A bucket case that highlights a span in text along with a label.
    :param true_label: The actual label
    :param predicted_label: The label that is predicted
    """

    true_label: str
    predicted_label: str


@dataclass
class BucketPerformance:
    bucket_interval: tuple
    n_samples: float
    bucket_samples: list[Any] = field(default_factory=list)
    performances: list[Performance] = field(default_factory=list)

    @classmethod
    def dict_conv(cls, k: str, v: dict):
        if k == 'performances':
            return [Performance.from_dict(v1) for v1 in v]
        if k == 'bucket_samples' and isinstance(v, dict):
            return [BucketCase.from_dict(v1) for v1 in v]
        else:
            return v

    @classmethod
    def from_dict(cls, data_dict: dict) -> BucketPerformance:
        field_names = set(f.name for f in dataclasses.fields(cls))
        return cls(
            **{k: cls.dict_conv(k, v) for k, v in data_dict.items() if k in field_names}
        )


@dataclass
class BucketCaseCollection:
    interval: tuple
    samples: list[BucketCase]

    def __len__(self):
        return len(self.samples)


@dataclass
class Result:
    overall: Optional[dict[str, Performance]] = None
    # {feature_name: {bucket_name: performance}}
    fine_grained: Optional[dict[str, list[BucketPerformance]]] = None
    calibration: Optional[list[Performance]] = None

    @classmethod
    def dict_conv(cls, k: str, v: dict):
        if k == 'overall':
            return {k1: Performance.from_dict(v1) for k1, v1 in v.items()}
        elif k == 'fine_grained':
            return {
                k1: [BucketPerformance.from_dict(v2) for v2 in v1]
                for k1, v1 in v.items()
            }
        elif k == 'calibration':
            return None if v is None else [Performance.from_dict(v1) for v1 in v]
        else:
            raise NotImplementedError

    @classmethod
    def from_dict(cls, data_dict: dict) -> Result:
        field_names = set(f.name for f in dataclasses.fields(cls))
        return cls(
            **{k: cls.dict_conv(k, v) for k, v in data_dict.items() if k in field_names}
        )


@dataclass
class SysOutputInfo:
    """Information about a system output and its analysis settings

    Attributes:
        task_name (str): the name of the task.
        system_name (str): the name of the system .
        dataset_name (str): the dataset used of the system.
        sub_dataset_name (str): the name of the sub-dataset.
        source_language (str): the language of the input
        target_language (str): the language of the output
        code (str): the url of the code.
        download_link (str): the url of the system output.
        paper (Paper, optional): the published paper of the system.
        features (Features, optional): the features used to describe system output's
            column type.
        is_print_case (bool): Whether or not to print out cases
        conf_value (float): The p-value of the confidence interval
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
    is_print_case: bool = True
    conf_value: float = 0.05
    system_details: Optional[dict] = None
    metric_configs: Optional[list[MetricConfig]] = None
    source_tokenizer: Optional[Tokenizer] = None
    target_tokenizer: Optional[Tokenizer] = None

    # set later
    # code: str = None
    # download_link: str = None
    # paper_info: PaperInfo = PaperInfo()
    features: Optional[Features] = None
    results: Result = field(default_factory=lambda: Result())

    def to_dict(self) -> dict:
        ret_dict = {}
        for f in dataclasses.fields(self):
            obj = getattr(self, f.name)
            if obj is not None:
                ret_dict[f.name] = general_to_dict(obj)
        return ret_dict

    def replace_nonstring_keys(self, data):
        if isinstance(data, list):
            for value in data:
                if isinstance(value, dict):
                    self.replace_nonstring_keys(value)
        else:
            replace_keys = []
            for key, value in data.items():
                if not isinstance(key, str):
                    replace_keys.append(key)
                if isinstance(value, dict) or isinstance(value, list):
                    self.replace_nonstring_keys(value)
            for key in replace_keys:
                data[str(key)] = data[key]
                del data[key]

    def write_to_directory(self, dataset_info_dir, file_name=""):
        """Write `SysOutputInfo` as JSON to `dataset_info_dir`."""
        file_path = (
            os.path.join(dataset_info_dir, config.SYS_OUTPUT_INFO_FILENAME)
            if file_name == ""
            else os.path.join(dataset_info_dir, file_name)
        )
        with open(file_path, "wb") as f:
            self._dump_info(f)

    def print_as_json(self, file=None):
        if file is None:
            file = sys.stdout
        data_dict = self.to_dict()
        self.replace_nonstring_keys(data_dict)
        try:
            json.dump(data_dict, fp=file, indent=2, default=lambda x: x.json_repr())
        except TypeError as e:
            raise e

    def _dump_info(self, file):
        """SystemOutputInfo => JSON"""
        data_dict = self.to_dict()
        self.replace_nonstring_keys(data_dict)
        file.write(
            json.dumps(data_dict, indent=2, default=lambda x: x.json_repr()).encode(
                "utf-8"
            )
        )

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
    def dict_conv(cls, k: str, v: dict):
        if k == 'results':
            return Result.from_dict(v)
        elif k.endswith('tokenizer'):
            return Tokenizer.from_dict(v)
        else:
            return v

    @classmethod
    def from_dict(cls, data_dict: dict) -> SysOutputInfo:
        field_names = set(f.name for f in dataclasses.fields(cls))
        return cls(
            **{k: cls.dict_conv(k, v) for k, v in data_dict.items() if k in field_names}
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
    metric_stats: list[MetricStats]
    active_features: list[str]


def print_bucket_perfs(bucket_perfs: list[BucketPerformance], print_information: str):
    metric_names = [x.metric_name for x in bucket_perfs[0].performances]
    for i, metric_name in enumerate(metric_names):
        get_logger('report').info(f"the information of #{print_information}#")
        get_logger('report').info(f"bucket_interval\t{metric_name}\t#samples")
        for bucket_perf in bucket_perfs:
            get_logger('report').info(
                f"{bucket_perf.bucket_interval}\t"
                f"{bucket_perf.performances[i].value}\t"
                f"{bucket_perf.n_samples}"
            )
        get_logger('report').info('')
