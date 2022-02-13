from dataclasses import dataclass, asdict, field
from typing import Any, List, Optional
from explainaboard.feature import Features
import json
import os
from explainaboard import config
from explainaboard.utils.logging import get_logger
import dataclasses
import copy

logger = get_logger(__name__)




@dataclass
class Table:
    # def __init__(self,
    #              table_iterator):
    #     self.table = []
    #     for _id, dict_features in table_iterator:
    #         self.table.append(dict_features)
    table: dict = None

    # def __post_init__(self):



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
    metric_name: float = None
    value: float = None
    confidence_score_low: float = None
    confidence_score_up:float = None

@dataclass
class BucketPerformance(Performance):
    bucket_name:str = None
    n_samples:float = None
    bucket_samples: Any = None


@dataclass
class Result:
    overall:Any = None
    calibration: List[Performance] = None
    fine_grained:Any = None
    is_print_case:bool = True
    is_print_confidence_interval:bool = True



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
    """

    # set in the system_output scripts
    task_name: str
    model_name: Optional[str] = None
    dataset_name: Optional[str] = None
    metric_names: Optional[List[str]] = None
    # language : str = "English"

    # set later
    # code: str = None
    # download_link: str = None
    # paper_info: PaperInfo = PaperInfo()
    features: Features = None
    results: Result = field(default_factory=lambda: Result())


    def write_to_directory(self, dataset_info_dir):
        """Write `SysOutputInfo` as JSON to `dataset_info_dir`.
        """
        with open(os.path.join(dataset_info_dir, config.SYS_OUTPUT_INFO_FILENAME), "wb") as f:
            self._dump_info(f)
    
    def to_dict(self) -> dict:
        return asdict(self)

    def to_memory(self):
        print(json.dumps(self.to_dict(), indent=4))


    def _dump_info(self, file):
        """SystemOutputInfo => JSON"""
        file.write(json.dumps(self.to_dict(), indent=4).encode("utf-8"))


    @classmethod
    def from_directory(cls, sys_output_info_dir: str) -> "SysOutputInfo":
        """Create SysOutputInfo from the JSON file in `sys_output_info_dir`.
        Args:
            sys_output_info_dir (`str`): The directory containing the metadata file. This
                should be the root directory of a specific dataset version.
        """
        logger.info("Loading Dataset info from %s", sys_output_info_dir)
        if not sys_output_info_dir:
            raise ValueError("Calling DatasetInfo.from_directory() with undefined dataset_info_dir.")

        with open(os.path.join(sys_output_info_dir, config.SYS_OUTPUT_INFO_FILENAME), "r", encoding="utf-8") as f:
            sys_output_info_dict = json.load(f)
        return cls.from_dict(sys_output_info_dict)

    # @classmethod
    # def from_dict(cls, task_name: str, sys_output_info_dict: dict) -> "SysOutputInfo":
    #     field_names = set(f.name for f in dataclasses.fields(cls))
    #     return cls(task_name, **{k: v for k, v in sys_output_info_dict.items() if k in field_names})

    @classmethod
    def from_dict(cls, sys_output_info_dict: dict) -> "SysOutputInfo":
        field_names = set(f.name for f in dataclasses.fields(cls))
        return cls(**{k: v for k, v in sys_output_info_dict.items() if k in field_names})


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