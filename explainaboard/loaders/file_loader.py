from __future__ import annotations

from collections.abc import Callable
import csv
from dataclasses import dataclass
from io import StringIO
import json
from typing import Any, cast, final, Optional, Type, Union

from datalabs import load_dataset
from datalabs.features.features import ClassLabel, Sequence

from explainaboard.constants import Source
from explainaboard.utils.typing_utils import narrow

DType = Union[Type[int], Type[float], Type[str], Type[dict], Type[list]]


@dataclass
class FileLoaderField:
    """
    :param src_name: field name in the source file. use int for tsv column indices and
        use str for dict keys
    :param target_name: field name expected in the loaded data
    :param dtype: data type of the field in the loaded data. It is only intended for
        simple type conversion so it only supports int, float and str. Pass in None
        to turn off type conversion.
    :param strip_before_parsing: call strip() on strings before casting to either str,
        int or float. It is only intended to be used with these three data types.
            It defaults to True for str. For all other types, it defaults to False
    :param parser: a custom parser for the field. When called,
        `data_points[idx][src_name]` is passed in as input, it is expected to return
        the parsed result. If parser is not None, `strip_before_parsing` and dtype
        will not have any effect.
    """

    src_name: Union[int, str]
    target_name: str
    dtype: Optional[DType] = None
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
        if self.dtype not in (str, int, float, dict, list, None):
            raise ValueError(
                "dtype must be one of str, int, float, dict, list, and None"
            )


class FileLoader:
    def __init__(
        self,
        fields: list[FileLoaderField] = None,
        use_idx_as_id: bool = True,
        id_field_name: Optional[str] = None,
    ) -> None:
        """Loader that loads data according to fields

        :param use_idx_as_id: whether to use sample indices as IDs. Generated IDs are
        str even though it represents an index. (This is to make sure all sample IDs
        are str.)
        """
        self._fields = fields or []
        self._use_idx_as_id = use_idx_as_id
        self._id_field_name = id_field_name

        self.validate()

    def validate(self):
        """validates fields"""
        if self._use_idx_as_id and self._id_field_name:
            raise ValueError("id_field_name must be None when use_idx_as_id is True")
        src_names = [field.src_name for field in self._fields]
        target_names = [field.target_name for field in self._fields]
        if len(src_names) != len(set(src_names)):
            raise ValueError("src_name must be unique")
        if len(target_names) != len(set(target_names)):
            raise ValueError("target_name must be unique")

    @final
    def add_fields(self, fields: list[FileLoaderField]):
        self._fields.extend(fields)
        self.validate()

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
        elif dtype == list or dtype == dict:
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

    def load_raw(self, data: str | DatalabLoaderOption, source: Source) -> list:
        """Load data from source and return an iterable of data points. It does not use
        fields information to parse the data points.

        :param data (str|DatalabLoaderOption): if str, it's either base64 encoded system
            output or a path
        :param source: source of data
        """
        raise NotImplementedError(
            "load_raw() is not implemented for the base FileLoader"
        )

    def load(self, data: str | DatalabLoaderOption, source: Source) -> list[dict]:
        """Load data from source, parse data points with fields information and return an
        iterable of data points.
        """
        raw_data = self.load_raw(data, source)
        parsed_data_points: list[dict] = []

        for idx, data_point in enumerate(raw_data):
            parsed_data_point = {}

            for field in self._fields:  # parse data point according to fields
                if (
                    isinstance(data_point, list)
                    and int(field.src_name) >= len(data_point)
                ) or (
                    isinstance(data_point, dict) and field.src_name not in data_point
                ):
                    cls = type(self).__name__
                    raise ValueError(
                        f'{cls} loading {data}: Could not find field '
                        f'"{field.src_name}" in datapoint {data_point}'
                    )
                parsed_data_point[field.target_name] = self.parse_data(
                    data_point[field.src_name], field
                )

            self.generate_id(parsed_data_point, idx)
            parsed_data_points.append(parsed_data_point)
        return parsed_data_points


class TSVFileLoader(FileLoader):
    def validate(self):
        super().validate()
        for field in self._fields:
            if not isinstance(field.src_name, int):
                raise ValueError("field src_name for TSVFileLoader must be an int")

    def load_raw(
        self, data: str | DatalabLoaderOption, source: Source
    ) -> list[list[str]]:
        data = narrow(data, str)
        if source == Source.in_memory:
            file = StringIO(data)
            lines = list(csv.reader(file, delimiter='\t', quoting=csv.QUOTE_NONE))
        elif source == Source.local_filesystem:
            with open(data, "r", encoding="utf8") as fin:
                lines = list(csv.reader(fin, delimiter='\t', quoting=csv.QUOTE_NONE))
        else:
            raise NotImplementedError
        return list(filter(lambda line: line, lines))  # remove empty lines


class CoNLLFileLoader(FileLoader):
    def __init__(self, fields: list[FileLoaderField] = None) -> None:
        super().__init__(fields, False)

    def validate(self):
        super().validate()
        if len(self._fields) not in [1, 2]:
            raise ValueError(
                "CoNLL file loader expects 1 or 2 fields "
                + f"({len(self._fields)} given)"
            )

    def load_raw(self, data: str | DatalabLoaderOption, source: Source) -> list[str]:
        data = narrow(data, str)
        if source == Source.in_memory:
            return data.splitlines()
        elif source == Source.local_filesystem:
            with open(data, "r", encoding="utf8") as fin:
                return [line.strip() for line in fin]
        raise NotImplementedError

    def load(self, data: str | DatalabLoaderOption, source: Source) -> list[dict]:
        raw_data = self.load_raw(data, source)
        parsed_samples: list[dict] = []
        guid = 0
        curr_sentence_fields: dict[Union[str, int], list[str]] = {
            field.src_name: [] for field in self._fields
        }

        def add_sample():
            nonlocal guid, curr_sentence_fields
            # uses the first field to check if data is empty
            if curr_sentence_fields.get(self._fields[0].src_name):
                new_sample: dict = {}
                for field in self._fields:  # parse data point according to fields
                    new_sample[field.target_name] = curr_sentence_fields[field.src_name]
                new_sample["id"] = str(guid)
                parsed_samples.append(new_sample)
                guid += 1
                curr_sentence_fields = {
                    field.src_name: [] for field in self._fields
                }  # reset

        max_field: int = max([narrow(x.src_name, int) for x in self._fields])
        for line in raw_data:
            # at sentence boundary
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                add_sample()
            else:
                splits = line.split("\t")
                if len(splits) <= max_field:  # not separated by tabs
                    splits = line.split(" ")
                if len(splits) <= max_field:  # not separated by tabs or spaces
                    raise ValueError(
                        f"not enough fields for {line} (sentence index: {guid})"
                    )

                for field in self._fields:
                    curr_sentence_fields[field.src_name].append(
                        self.parse_data(splits[narrow(field.src_name, int)], field)
                    )

        add_sample()  # add last example
        return parsed_samples


class JSONFileLoader(FileLoader):
    def load_raw(self, data: str | DatalabLoaderOption, source: Source) -> Any:
        data = narrow(data, str)
        if source == Source.in_memory:
            return json.loads(data)
        elif source == Source.local_filesystem:
            with open(data, 'r', encoding="utf8") as json_file:
                return json.load(json_file)
        raise NotImplementedError


@dataclass
class DatalabLoaderOption:
    dataset: str
    subdataset: str | None = None
    split: str = "test"


class DatalabFileLoader(FileLoader):
    @classmethod
    def replace_one(cls, names: list[str], lab: int):
        return names[lab] if lab != -1 else '_NULL_'

    @classmethod
    def replace_labels(cls, features: dict, example: dict) -> dict:
        new_example = {}
        for examp_k, examp_v in example.items():
            examp_f = features[examp_k]
            # Label feature
            if isinstance(examp_f, ClassLabel):
                names = cast(ClassLabel, examp_f).names
                new_example[examp_k] = cls.replace_one(names, examp_v)
            # Sequence feature
            elif isinstance(examp_f, Sequence):
                examp_seq = cast(Sequence, examp_f)
                # Sequence of labels
                if isinstance(examp_seq.feature, ClassLabel):
                    names = cast(ClassLabel, examp_seq.feature).names
                    new_example[examp_k] = [cls.replace_one(names, x) for x in examp_v]
                # Sequence of anything else
                else:
                    new_example[examp_k] = examp_v
            # Anything else
            else:
                new_example[examp_k] = examp_v
        return new_example

    def load_raw(self, data: str | DatalabLoaderOption, source: Source) -> list[dict]:
        config = narrow(data, DatalabLoaderOption)
        dataset = load_dataset(
            config.dataset, config.subdataset, split=config.split, streaming=False
        )
        return [self.replace_labels(dataset.info.features, x) for x in dataset]


class TextFileLoader(FileLoader):
    """loads a text file. Each line is a different sample.
    - only one field is allowed. It is often used for predicted outputs.
    """

    def __init__(
        self,
        target_name: str = "output",
        dtype: DType = str,
        strip_before_parsing: Optional[bool] = None,
    ) -> None:
        # src_name is not used for this file loader, it overrides the base load method.
        super().__init__(
            [FileLoaderField("_", target_name, dtype, strip_before_parsing)],
            use_idx_as_id=True,
        )

    @classmethod
    def load_raw(cls, data: str | DatalabLoaderOption, source: Source) -> list:
        data = narrow(data, str)
        if source == Source.in_memory:
            return data.splitlines()
        elif source == Source.local_filesystem:
            with open(data, "r", encoding="utf8") as f:
                return f.readlines()
        raise NotImplementedError

    def validate(self):
        super().validate()
        if len(self._fields) != 1:
            raise ValueError("Text File Loader only takes one field")

    def load(self, data: str | DatalabLoaderOption, source: Source) -> list[dict]:
        raw_data = self.load_raw(data, source)
        parsed_data_points: list[dict] = []

        for idx, data_point in enumerate(raw_data):
            parsed_data_point = {}
            field = self._fields[0]

            parsed_data_point[field.target_name] = self.parse_data(data_point, field)
            self.generate_id(parsed_data_point, idx)
            parsed_data_points.append(parsed_data_point)
        return parsed_data_points
