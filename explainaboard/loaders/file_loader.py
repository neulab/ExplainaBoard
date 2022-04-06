from __future__ import annotations

from collections.abc import Callable, Iterable
import csv
from dataclasses import dataclass
from io import StringIO
import json
from typing import Any, Optional, Union

from explainaboard.constants import Source


@dataclass
class FileLoaderField:
    """
    Args:
        src_name: field name in the source file. use int for tsv column indices and use
            str for dict keys
        target_name: field name expected in the loaded data
        dtype: data type of the field in the loaded data. It is only intended for simple
            type conversion so it only supports int, float and str. Pass in None to turn
            off type conversion.
        strip_before_parsing: call strip() on strings before casting to either str, int
            or float. It is only intended to be used with these three data types.
            It defaults to True for str. For all other types, it defaults to False
        parser: a custom parser for the field. When called, `data_points[idx][src_name]`
            is passed in as input, it is expected to return the parsed result.
            If parser is not None, `strip_before_parsing` and dtype will not have any
            effect.
    """

    src_name: Union[int, str]
    target_name: str
    dtype: Optional[Union[type[int], type[float], type[str], type[dict]]] = None
    strip_before_parsing: Optional[bool] = None
    parser: Optional[Callable] = None

    def __post_init__(self):
        if self.strip_before_parsing is None:
            self.strip_before_parsing = self.dtype == str

        # validation
        for name in (self.src_name, self.target_name):
            if not isinstance(name, str) and not isinstance(name, int):
                raise ValueError("src_name and target_name must be str or int")

        if self.dtype is None and self.strip_before_parsing:
            raise ValueError(
                "strip_before_parsing only works with int, float and str types"
            )
        if self.dtype not in (str, int, float, dict, None):
            raise ValueError("dtype must be one of str, int, float, dict and None")


class FileLoader:
    def __init__(
        self,
        fields: list[FileLoaderField] = None,
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
            data = (
                data.strip() if isinstance(data, str) else data
            )  # some time data could be a nested json object
        dtype = field.dtype
        if dtype == int:
            return int(data)
        elif dtype == float:
            return float(data)
        elif dtype == str:
            return str(data)
        elif dtype == dict:
            return data  # TODO(Pengfei): I add the `dict` type for temporal use,
            # but wonder if we need to generalize the current type mechanism,
        elif dtype is None:
            return data
        raise NotImplementedError(f"dtype {dtype} is not supported")

    def generate_id(self, parsed_data_point: dict, sample_idx: int):
        """generates an id attribute for each data point in place"""
        if self._use_idx_as_id:
            parsed_data_point["id"] = str(sample_idx)
        elif self._id_field_name:
            if self._id_field_name not in parsed_data_point:
                raise ValueError(
                    f"The {sample_idx} data point in system outputs file does not have "
                    f"field {self._id_field_name}"
                )
            parsed_data_point["id"] = str(parsed_data_point[self._id_field_name])

    @classmethod
    def load_raw(cls, data: str, source: Source) -> Iterable:
        """Load data from source and return an iterable of data points. It does not use
        fields information to parse the data points.

        Args:
            data (str): base64 encoded system output content or a path for the system
                output file
            source: source of data
        """
        raise NotImplementedError(
            "load_raw() is not implemented for the base FileLoader"
        )

    def load(
        self, data: str, source: Source, user_defined_features_configs: dict
    ) -> Iterable[dict]:
        """Load data from source, parse data points with fields information and return an
        iterable of data points.
        """
        raw_data = self.load_raw(data, source)
        parsed_data_points: list[dict] = []

        for idx, data_point in enumerate(raw_data):
            parsed_data_point = {}

            for field in self._fields:  # parse data point according to fields
                parsed_data_point[field.target_name] = self.parse_data(
                    data_point[field.src_name], field
                )

            self.generate_id(parsed_data_point, idx)
            parsed_data_points.append(parsed_data_point)
        return parsed_data_points


class TSVFileLoader(FileLoader):
    def __init__(
        self,
        fields: list[FileLoaderField] = None,
        use_idx_as_id: bool = True,
        id_field_name: Optional[str] = None,
    ) -> None:
        super().__init__(fields, use_idx_as_id, id_field_name)
        for field in self._fields:
            if not isinstance(field.src_name, int):
                raise ValueError("field src_name for TSVFileLoader must be an int")

    @classmethod
    def load_raw(cls, data: str, source: Source) -> Iterable:
        if source == Source.in_memory:
            file = StringIO(data)
            return csv.reader(file, delimiter='\t')
        elif source == Source.local_filesystem:
            content: list[list[str]] = []
            with open(data, "r", encoding="utf8") as fin:
                for record in csv.reader(fin, delimiter='\t'):
                    content.append(record)
            return content
        raise NotImplementedError


class CoNLLFileLoader(FileLoader):
    def __init__(self) -> None:
        super().__init__(fields=[], use_idx_as_id=False, id_field_name=None)

    @classmethod
    def load_raw(cls, data: str, source: Source) -> Iterable:
        if source == Source.in_memory:
            return data.splitlines()
        elif source == Source.local_filesystem:
            content = []
            with open(data, "r", encoding="utf8") as fin:
                for record in fin:
                    content.append(record)
            return content
        raise NotImplementedError

    def load(
        self, data: str, source: Source, user_defined_features_configs: dict
    ) -> Iterable[dict]:
        raw_data = self.load_raw(data, source)
        parsed_data_points: list[dict] = []
        guid = 0
        tokens: list[str] = []
        ner_true_tags: list[str] = []
        ner_pred_tags: list[str] = []

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
    def load_raw(cls, data: str, source: Source) -> Iterable:
        if source == Source.in_memory:
            return json.loads(data)
        elif source == Source.local_filesystem:
            with open(data, 'r', encoding="utf8") as json_file:
                data = json_file.read()
                return json.loads(data)
        raise NotImplementedError

    def load(
        self, data: str, source: Source, user_defined_features_configs: dict
    ) -> Iterable[dict]:
        raw_data = self.load_raw(data, source)
        parsed_data_points: list[dict] = []

        for idx, data_point in enumerate(raw_data):
            parsed_data_point = {}

            for field in self._fields:  # parse data point according to fields
                parsed_data_point[field.target_name] = self.parse_data(
                    data_point[field.src_name], field
                )

            # add idx as the sample id
            self.generate_id(parsed_data_point, idx)

            if (
                user_defined_features_configs is not None
                and len(user_defined_features_configs) != 0
            ):
                # additional user-defined features
                parsed_data_point.update(
                    {
                        feature_name: data_point[feature_name]
                        for feature_name in user_defined_features_configs
                    }
                )
            parsed_data_points.append(parsed_data_point)

        return parsed_data_points


class DatalabFileLoader(FileLoader):
    @classmethod
    def load_raw(cls, data: str, source: Source) -> Iterable:
        if source == Source.in_memory:
            return data
        raise NotImplementedError
