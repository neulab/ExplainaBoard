from __future__ import annotations
import csv
from ctypes import Union
from dataclasses import dataclass
from io import StringIO
import json
from typing import Any, Callable, Iterable, List, Optional

from enum import Enum

from explainaboard.constants import Source


class FileLoader:
    def __init__(
        self,
        fields: List[FileLoaderField] = None,
        use_idx_as_id: bool = True,
        id_field_name: Optional[str] = None,
    ) -> None:
        self._fields = fields or []
        self._use_idx_as_id = use_idx_as_id
        self._id_field_name = id_field_name

        # validations
        if self._use_idx_as_id and self._id_field_name:
            raise ValueError("id_field_name must be None when use_idx_as_id is True")
        src_names = [field.src_name for field in self._fields]
        target_names = [field.target_name for field in self._fields]
        if len(src_names) != len(set(src_names)):
            raise ValueError("src_name must be unique")
        if len(target_names) != len(set(target_names)):
            raise ValueError("target_name must be unique")

    @staticmethod
    def parse_data(data: Any, field: FileLoaderField) -> Any:
        if field.parser:
            return field.parser(data)
        if field.strip_before_parsing:
            data = data.strip()
        dtype = field.dtype
        if dtype == FileLoaderDType.int:
            return int(data)
        if dtype == FileLoaderDType.float:
            return float(data)
        if dtype == FileLoaderDType.str:
            return str(data)
        if dtype == FileLoaderDType.other:
            return data

    def generate_id(self, parsed_data_point: dict, sample_idx: int):
        """generates an id attribute for each data point in place"""
        if self._use_idx_as_id:
            parsed_data_point["id"] = str(sample_idx)
        elif self._id_field_name:
            if self._id_field_name not in parsed_data_point:
                raise ValueError(
                    f"The {sample_idx} data point in system outputs file does not have field {self._id_field_name}"
                )
            parsed_data_point["id"] = str(parsed_data_point[self._id_field_name])

    @classmethod
    def load_raw(self, data: str, source: Source) -> Iterable:
        """Load data from source and return an iterable of data points. It does not use
        fields information to parse the data points.

        Args:
            data (str): base64 encoded system output content or a path for the system output file
                source: source of data
        """
        raise NotImplementedError(
            "load_raw() is not implemented for the base FileLoader"
        )

    def load(self, data: str, source: Source) -> Iterable[dict]:
        """Load data from source, parse data points with fields information and return an
        iterable of data points.
        """
        raw_data = self.load_raw(data, source)
        parsed_data_points: List[dict] = []
        for idx, data_point in enumerate(raw_data):
            parsed_data_point = {}

            for field in self._fields:  # parse data point according to fields
                parsed_data_point[field.target_name] = self.parse_data(
                    data_point[field.src_name], field
                )

            self.generate_id(parsed_data_point, idx)
            parsed_data_points.append(parsed_data_point)
        return parsed_data_points


@dataclass
class FileLoaderField:
    src_name: Union[int, str]  # int for tsv column indices and string for dict keys
    target_name: str
    dtype: FileLoaderDType
    strip_before_parsing: bool = False
    parser: Callable = None

    def __post_init__(self):
        for name in (self.src_name, self.target_name):
            if not isinstance(name, str) and not isinstance(name, int):
                raise ValueError("src_name and target_name must be str or int")
        if self.dtype == FileLoaderDType.other and self.strip_before_parsing:
            raise ValueError("dict type field does not support strip_before_parsing")


class FileLoaderDType(str, Enum):
    int = "int"
    float = "float"
    str = "str"
    other = "other"


class TSVFileLoader(FileLoader):
    def __init__(
        self,
        fields: List[FileLoaderField] = None,
        use_idx_as_id: bool = True,
        id_field_name: Optional[str] = None,
    ) -> None:
        super().__init__(fields, use_idx_as_id, id_field_name)
        for field in self._fields:
            if not isinstance(field.src_name, int):
                raise ValueError("field src_name for TSVFileLoader must be an int")
            if field.dtype == FileLoaderDType.other:
                raise ValueError("TSVFilerLoader doesn't support 'other' field type")

    @classmethod
    def load_raw(self, data: str, source: Source) -> Iterable:
        if source == Source.in_memory:
            file = StringIO(data)
            return csv.reader(file, delimiter='\t')
        if source == Source.local_filesystem:
            content: List[str] = []
            with open(data, "r", encoding="utf8") as fin:
                for record in csv.reader(fin, delimiter='\t'):
                    content.append(record)
            return content
        raise NotImplementedError


class CoNLLFilerLoader(FileLoader):
    def __init__(self) -> None:
        super().__init__(fields=[], use_idx_as_id=False, id_field_name=False)

    @classmethod
    def load_raw(self, data: str, source: Source) -> Iterable:
        if source == Source.in_memory:
            return data.splitlines()
        if source == Source.local_filesystem:
            content = []
            with open(data, "r", encoding="utf8") as fin:
                for record in fin:
                    content.append(record)
            return content
        raise NotImplementedError

    def load(self, data: str, source: Source) -> Iterable[dict]:
        raw_data = self.load_raw(data, source)
        parsed_data_points: List[dict] = []
        guid = 0
        tokens = []
        ner_true_tags = []
        ner_pred_tags = []

        for id, line in enumerate(raw_data):
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if tokens:
                    parsed_data_points.append(
                        {
                            "id": str(guid),
                            "tokens": tokens,
                            "true_tags": ner_true_tags,
                            "pred_tags": ner_pred_tags,
                        }
                    )
                    guid += 1
                    tokens = []
                    ner_true_tags = []
                    ner_pred_tags = []
            else:
                splits = (
                    line.split("\t") if len(line.split("\t")) == 3 else line.split(" ")
                )
                tokens.append(splits[0].strip())
                ner_true_tags.append(splits[1].strip())
                ner_pred_tags.append(splits[2].strip())

        # last example
        parsed_data_points.append(
            {
                "id": str(guid),
                "tokens": tokens,
                "true_tags": ner_true_tags,
                "pred_tags": ner_pred_tags,
            }
        )
        return parsed_data_points


class JSONFileLoader(FileLoader):
    @classmethod
    def load_raw(self, data: str, source: Source) -> Iterable:
        if source == Source.in_memory:
            return json.loads(data)
        if source == Source.local_filesystem:
            with open(data, 'r', encoding="utf8") as json_file:
                data = json_file.read()
                return json.loads(data)
        raise NotImplementedError


class DatalabFileLoader(FileLoader):
    @classmethod
    def load_raw(self, data: str, source: Source) -> Iterable:
        if source == Source.in_memory:
            return data
        raise NotImplementedError
