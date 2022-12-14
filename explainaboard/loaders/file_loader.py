"""File loaders."""

from __future__ import annotations

from collections.abc import Callable, Iterable
import copy
import csv
import dataclasses
from dataclasses import dataclass, field
from io import StringIO
import itertools
import json
from typing import Any, cast, ClassVar, final, Optional, Sized, TypeVar, Union

from datalabs import DatasetDict, IterableDatasetDict, load_dataset
from datalabs.features.features import ClassLabel, Sequence

from explainaboard.analysis.analyses import Analysis
from explainaboard.analysis.feature import FeatureType
from explainaboard.constants import Source
from explainaboard.serialization.serializers import PrimitiveSerializer
from explainaboard.utils.load_resources import get_customized_features
from explainaboard.utils.typing_utils import narrow

DType = Union[type[int], type[float], type[str], type[dict], type[list]]
T = TypeVar("T")


@dataclass
class FileLoaderField:
    """A filed in a file loader.

    Attributes:
        src_name: field name in the source file. use int for tsv column indices,
            str for dict keys, or tuple for hierarchical dict keys
        target_name: field name expected in the loaded data
        dtype: data type of the field in the loaded data. It is only intended for
            simple type conversion so it only supports int, float and str. Pass in None
            to turn off type conversion.
        strip_before_parsing: call strip() on strings before casting to either str,
            int or float. It is only intended to be used with these three data types.
                It defaults to True for str. For all other types, it defaults to False
        parser: a custom parser for the field. When called,
            `data_points[idx][src_name]` is passed in as input, it is expected to return
            the parsed result. If parser is not None, `strip_before_parsing` and dtype
            will not have any effect.
    """

    src_name: int | str | Iterable[str]
    target_name: int | str
    dtype: Optional[DType] = None
    strip_before_parsing: Optional[bool] = None
    parser: Optional[Callable[[str], str]] = None
    optional: bool = False

    # Special constants used in field mapping
    SOURCE_LANGUAGE: ClassVar[str] = "__SOURCE_LANGUAGE__"
    TARGET_LANGUAGE: ClassVar[str] = "__TARGET_LANGUAGE__"

    def __post_init__(self):
        """Validate data and set defaults."""
        if self.strip_before_parsing is None:
            self.strip_before_parsing = self.dtype == str

        if self.dtype is None and self.strip_before_parsing:
            raise ValueError(
                "strip_before_parsing only works with int, float and str types"
            )
        if self.dtype not in (str, int, float, dict, list, None):
            raise ValueError(
                "dtype must be one of str, int, float, dict, list, and None"
            )


@dataclass
class FileLoaderMetadata:
    """Metadata populated in the process of loading the dataset or output files.

    Attributes:
        source_language: The language of the input
        target_language: The language of the output
        supported_languages: All languages supported by the dataset at all
        task: The specific task to be analyzed
        supported_tasks: The task or tasks that *can* be handled (e.g. by a dataset)
    """

    system_name: str | None = None
    dataset_name: str | None = None
    sub_dataset_name: str | None = None
    split: str | None = None
    source_language: str | None = None
    target_language: str | None = None
    supported_languages: list[str] = field(default_factory=list)
    task_name: str | None = None
    supported_tasks: list[str] = field(default_factory=list)
    custom_features: dict[str, dict[str, FeatureType]] = field(default_factory=dict)
    # analysis level name -> list of analyses dictionary
    custom_analyses: list[Analysis] = field(default_factory=list)

    def merge(self, other: FileLoaderMetadata) -> None:
        """Merge the information from the two pieces of metadata.

        In the case that the two conflict, the passed-in metadata get preference
        with the exception of `custom_features` and `custom_analyses`. For those
        two fields, self is updated/extended with the values from other.
        """
        # TODO(gneubig): This should be changed into a for loop
        self.system_name = other.system_name or self.system_name
        self.dataset_name = other.dataset_name or self.dataset_name
        self.sub_dataset_name = other.sub_dataset_name or self.sub_dataset_name
        self.split = other.split or self.split
        self.source_language = other.source_language or self.source_language
        self.target_language = other.target_language or self.target_language
        self.supported_languages = other.supported_languages or self.supported_languages
        self.task_name = other.task_name or self.task_name
        self.supported_tasks = other.supported_tasks or self.supported_tasks
        for level, features in other.custom_features.items():
            self.custom_features.setdefault(level, {})
            self.custom_features[level].update(features)
        self.custom_analyses.extend(other.custom_analyses)

    @classmethod
    def from_dict(cls, data: dict) -> FileLoaderMetadata:
        """Deserialize a dictionary into File Loader Metadata.

        Args:
            data: The data to serialize.
        """
        # TODO(gneubig): A better way to do this might be through a library such as
        #                pydantic or dacite
        source_language = data.get("source_language")
        target_language = data.get("target_language")
        if data.get("language"):
            if source_language or target_language:
                raise ValueError(
                    'can not set both "language" and "source_language"/'
                    '"target_language"'
                )
            source_language = target_language = data.get("language")
        custom_features: dict[str, dict[str, FeatureType]] = {}
        custom_analyses: list[Analysis] = []
        serializer = PrimitiveSerializer()
        if "custom_features" in data:
            custom_features = {
                k1: {
                    # See https://github.com/python/mypy/issues/4717
                    k2: narrow(FeatureType, serializer.deserialize(v2))  # type: ignore
                    for k2, v2 in v1.items()
                }
                for k1, v1 in data["custom_features"].items()
            }
        if "custom_analyses" in data:
            custom_analyses = [
                narrow(Analysis, serializer.deserialize(analysis))  # type: ignore
                for analysis in data["custom_analyses"]
            ]
        return FileLoaderMetadata(
            system_name=data.get("system_name"),
            dataset_name=data.get("dataset_name"),
            sub_dataset_name=data.get("sub_dataset_name"),
            split=data.get("split"),
            source_language=source_language,
            target_language=target_language,
            supported_languages=data.get("supported_languages", []),
            task_name=data.get("task_name"),
            supported_tasks=data.get("supported_tasks", []),
            custom_features=custom_features,
            custom_analyses=custom_analyses,
        )

    @classmethod
    def from_file(cls, file_name: str) -> FileLoaderMetadata:
        """Load meta data from a file.

        Args:
            file_name: The name of the file to load from.

        Returns:
            A file loader metadata class.
        """
        with open(file_name, "r") as file_in:
            my_data = json.load(file_in)
            if not isinstance(my_data, dict) or "metadata" not in my_data:
                raise ValueError(f"Could not find metadata in {file_name}")
            else:
                return FileLoaderMetadata.from_file(my_data["metadata"])


@dataclass
class FileLoaderReturn(Sized):
    """Data returned by a FileLoader.

    Attributes:
        samples: A list of samples from the dataset
        metadata: Metadata regarding the samples or the dataset
    """

    samples: list
    metadata: FileLoaderMetadata = dataclasses.field(
        default_factory=lambda: FileLoaderMetadata()
    )

    def __len__(self) -> int:
        """Return the length of the samples."""
        return len(self.samples)

    def __getitem__(self, item: int) -> Any:
        """Get a certain sample from the dataset."""
        return self.samples[item]


class FileLoader:
    """A class that loads raw data from a file."""

    def __init__(
        self,
        fields: list[FileLoaderField] = None,
        use_idx_as_id: bool = True,
        id_field_name: Optional[str] = None,
    ) -> None:
        """Loader that loads data according to fields.

        Args:
            fields: A specification of the fields to be read in from the file.
            use_idx_as_id: whether to use sample indices as IDs. Generated IDs are
                            str even though it represents an index.
                            (This is to make sure all sample IDs are str.)
            id_field_name: The name of the field to be used as an ID
        """
        self._fields = fields or []
        self._use_idx_as_id = use_idx_as_id
        self._id_field_name = id_field_name

        self.validate()

    def validate(self) -> None:
        """Validates the setting of the fields.

        Raises:
            ValueError: if the fields are not valid
        """
        if self._use_idx_as_id and self._id_field_name:
            raise ValueError("id_field_name must be None when use_idx_as_id is True")
        target_names = [field.target_name for field in self._fields]
        if len(target_names) != len(set(target_names)):
            raise ValueError("target_name must be unique")

    @final
    def add_fields(self, fields: list[FileLoaderField]) -> None:
        """Add more more fields to the FileLoader.

        Args:
            fields: The fields to add.

        Raises:
            ValueError: if the fields are wrong.
        """
        self._fields.extend(fields)
        self.validate()

    @staticmethod
    def parse_data(data: Any, field: FileLoaderField) -> Any:
        """Parse data loaded in from a file to the required data type.

        Args:
            data: The data loaded in from the file.
            field: Information about the field.

        Returns:
            The parsed data.
        """
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

    def generate_id(self, parsed_data_point: dict, sample_idx: int) -> None:
        """Generates an id attribute for each data point in place.

        Args:
            parsed_data_point: The data point parsed into a dict.
            sample_idx: The ID of the sample.
        """
        if self._use_idx_as_id:
            parsed_data_point["id"] = str(sample_idx)
        elif self._id_field_name:
            if self._id_field_name not in parsed_data_point:
                raise ValueError(
                    f"The {sample_idx} data point in system outputs file does not have "
                    f"field {self._id_field_name}"
                )
            parsed_data_point["id"] = str(parsed_data_point[self._id_field_name])

    def load_raw(
        self, data: str | DatalabLoaderOption, source: Source
    ) -> FileLoaderReturn:
        """Load data from source and return an iterable of data points.

        It does not use fields information to parse the data points.

        Args:
            data: if str, it's either base64 encoded system output or a path
            source: source of data

        Returns:
            The data loaded from the file.
        """
        raise NotImplementedError(
            "load_raw() is not implemented for the base FileLoader"
        )

    def _map_fields(self, fields: list, field_mapping: dict[str, str] | None = None):
        new_fields = copy.deepcopy(fields)
        if field_mapping is not None:
            for f in new_fields:
                if isinstance(f.src_name, str):
                    f.src_name = field_mapping.get(f.src_name, f.src_name)
                elif isinstance(f.src_name, Iterable):
                    f.src_name = [field_mapping.get(x, x) for x in f.src_name]
        return new_fields

    @classmethod
    def find_field(
        cls,
        data_point: dict,
        field: FileLoaderField,
        field_mapping: dict[str, str] | None = None,
    ):
        """In a structured dictionary, find a specified field.

        This can be specified by an
        * int index to a list (data_point[field])
        * str index to a dictionary (datapoint[field])
        * Iterable[str] index to a dictionary (datapoint[field[0]][field[1]]...)

        Args:
            data_point: The data to search
            field: The file loader field corresponding to the dict
            field_mapping: A mapping between field names. If a str in a field name
              exists as a key in the mapping, the value will be used to search instead

        Returns:
            the required data
        """
        if field_mapping is None:
            field_mapping = {}
        if isinstance(data_point, list):
            int_idx = int(field.src_name)
            if int_idx >= len(data_point):
                raise ValueError(
                    f"{cls.__name__}: Could not find "
                    f'field "{field.src_name}" in datapoint {data_point}'
                )
            return data_point[int_idx]
        elif isinstance(data_point, dict):
            if isinstance(field.src_name, int):
                raise ValueError(f"unexpected int index for dict data_point in {field}")
            # Parse a string or tuple identifier
            field_list = (
                [field.src_name] if isinstance(field.src_name, str) else field.src_name
            )
            ret_dict = data_point
            for sub_field in field_list:
                sub_field = field_mapping.get(sub_field, sub_field)
                if sub_field not in ret_dict:
                    if field.optional:
                        return None
                    raise ValueError(
                        f"{cls.__name__}: Could not find "
                        f'field "{field.src_name}" in datapoint {data_point}'
                    )
                ret_dict = ret_dict[sub_field]
            return ret_dict

    def load(
        self,
        data: str | DatalabLoaderOption,
        source: Source,
        field_mapping: dict[str, str] | None = None,
    ) -> FileLoaderReturn:
        """Load data from source, parse data points with fields information.

        Args:
            data: An indication of the data to be loading
            source: The source from which it should be loaded
            field_mapping: A mapping from field name in the loader spec to field name
              in the actual input

        Returns:
             an iterable of data points.
        """
        raw_data = self.load_raw(data, source)
        parsed_data_points: list[dict] = []

        # Get language information from meta-data if it doesn't exist already
        actual_mapping = field_mapping or {}
        for lang, meta in [
            (FileLoaderField.SOURCE_LANGUAGE, raw_data.metadata.source_language),
            (FileLoaderField.TARGET_LANGUAGE, raw_data.metadata.target_language),
        ]:
            temp = actual_mapping.get(lang) or meta
            if temp is not None:
                actual_mapping[lang] = temp

        # map the field names
        before_fields = copy.deepcopy(self._fields)
        fields = self._map_fields(self._fields, actual_mapping)
        if raw_data.metadata.custom_features is not None:
            for level_name, feats in raw_data.metadata.custom_features.items():
                if level_name == "example":
                    for feat in feats:
                        fields.append(FileLoaderField(feat, feat, None))
                else:
                    raise ValueError(
                        "cannot currently load custom features other "
                        f"than on the example level (got {level_name})"
                    )
        assert [x.src_name for x in before_fields] == [x.src_name for x in self._fields]

        # process the actual data
        for idx, data_point in enumerate(raw_data.samples):
            parsed_data_point = {}
            for f in fields:  # parse data point according to fields
                find_field_result = self.find_field(data_point, f, field_mapping)
                if find_field_result is not None:
                    parsed_data_point[f.target_name] = self.parse_data(
                        find_field_result, f
                    )
            self.generate_id(parsed_data_point, idx)
            parsed_data_points.append(parsed_data_point)
        return FileLoaderReturn(parsed_data_points, raw_data.metadata)


class TSVFileLoader(FileLoader):
    """A class for loading from TSV files."""

    def validate(self) -> None:
        """See FileLoader.validate."""
        super().validate()
        for f in self._fields:
            if not isinstance(f.src_name, int):
                raise ValueError("field src_name for TSVFileLoader must be an int")

    def load_raw(
        self, data: str | DatalabLoaderOption, source: Source
    ) -> FileLoaderReturn:
        """See FileLoader.load_raw."""
        data = narrow(str, data)
        if source == Source.in_memory:
            file = StringIO(data)
            lines = list(csv.reader(file, delimiter="\t", quoting=csv.QUOTE_NONE))
        elif source == Source.local_filesystem:
            with open(data, "r", encoding="utf8") as fin:
                lines = list(csv.reader(fin, delimiter="\t", quoting=csv.QUOTE_NONE))
        else:
            raise NotImplementedError
        return FileLoaderReturn(
            list(filter(lambda line: line, lines))
        )  # remove empty lines


class CoNLLFileLoader(FileLoader):
    """A loader from CoNLL-formatted files."""

    def __init__(self, fields: list[FileLoaderField] = None) -> None:
        """Constructor.

        Args:
            fields: A list of fields to read from each column.
        """
        super().__init__(fields, False)

    def validate(self) -> None:
        """See FileLoader.validate."""
        super().validate()
        if len(self._fields) not in [1, 2]:
            raise ValueError(
                "CoNLL file loader expects 1 or 2 fields "
                + f"({len(self._fields)} given)"
            )

    def load_raw(
        self, data: str | DatalabLoaderOption, source: Source
    ) -> FileLoaderReturn:
        """See FileLoader.load_raw."""
        data = narrow(str, data)
        if source == Source.in_memory:
            return FileLoaderReturn(data.splitlines())
        elif source == Source.local_filesystem:
            with open(data, "r", encoding="utf8") as fin:
                return FileLoaderReturn([line.strip() for line in fin])
        raise NotImplementedError

    def load(
        self,
        data: str | DatalabLoaderOption,
        source: Source,
        field_mapping: dict[str, str] | None = None,
    ) -> FileLoaderReturn:
        """See FileLoader.load."""
        raw_data = self.load_raw(data, source)
        parsed_samples: list[dict] = []
        guid = 0
        curr_sentence_fields: dict[str | int | Iterable[str], list[str]] = {
            field.src_name: [] for field in self._fields
        }

        def add_sample():
            nonlocal guid, curr_sentence_fields
            # uses the first field to check if data is empty
            if curr_sentence_fields.get(self._fields[0].src_name):
                new_sample: dict = {}
                for f in self._fields:  # parse data point according to fields
                    new_sample[f.target_name] = curr_sentence_fields[f.src_name]
                new_sample["id"] = str(guid)
                parsed_samples.append(new_sample)
                guid += 1
                curr_sentence_fields = {
                    field.src_name: [] for field in self._fields
                }  # reset

        max_field: int = max([narrow(int, x.src_name) for x in self._fields])
        for line in raw_data.samples:
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

                for f in self._fields:
                    curr_sentence_fields[f.src_name].append(
                        self.parse_data(splits[narrow(int, f.src_name)], f)
                    )

        add_sample()  # add last example
        return FileLoaderReturn(parsed_samples, metadata=raw_data.metadata)


class JSONFileLoader(FileLoader):
    """A loader from JSON files."""

    def load_raw(
        self, data: str | DatalabLoaderOption, source: Source
    ) -> FileLoaderReturn:
        """See FileLoader.load_raw."""
        data = narrow(str, data)
        if source == Source.in_memory:
            loaded = json.loads(data)
        elif source == Source.local_filesystem:
            with open(data, "r", encoding="utf8") as json_file:
                loaded = json.load(json_file)
        else:
            raise NotImplementedError
        if isinstance(loaded, list):
            return FileLoaderReturn(loaded)
        else:
            if "examples" not in loaded or not isinstance(loaded["examples"], list):
                raise ValueError(
                    f"Error loading {data}. Input JSON files in dict "
                    'format must have a list "examples"'
                )
            raw_data = loaded.pop("examples")
            if len(loaded) > 1 or (len(loaded) == 1 and "metadata" not in loaded):
                raise ValueError(
                    f"Error loading {data}. Input JSON files in dict "
                    'format must have "examples" and optionally '
                    '"metadata", nothing else'
                )
            metadata = FileLoaderMetadata.from_dict(loaded.get("metadata", {}))
            return FileLoaderReturn(raw_data, metadata=metadata)


@dataclass
class DatalabLoaderOption:
    """A class representing the options when using DataLabLoader.

    Attributes:
        dataset: The name of the dataset
        subdataset: The name of the sub dataset (optional)
        split: The name of the split
        custom_features: Custom features that are added to the retrieved file
        custom_analyses: Custom analyses that are added
    """

    dataset: str
    subdataset: str | None = None
    split: str = "test"
    custom_features: dict[str, dict[str, FeatureType]] = field(default_factory=dict)
    custom_analyses: list[Analysis] = field(default_factory=list)


class DatalabFileLoader(FileLoader):
    """A file loader for loading from DataLab."""

    @classmethod
    def _replace_one(cls, names: list[str], lab: int):
        return names[lab] if lab != -1 else "_NULL_"

    @classmethod
    def _replace_labels(cls, features: dict, example: dict) -> dict:
        new_example = {}
        for examp_k, examp_v in example.items():
            examp_f = features[examp_k]
            # Label feature
            if isinstance(examp_f, ClassLabel):
                names = cast(ClassLabel, examp_f).names
                new_example[examp_k] = cls._replace_one(names, examp_v)
            # Sequence feature
            elif isinstance(examp_f, Sequence):
                examp_seq = cast(Sequence, examp_f)
                # Sequence of labels
                if isinstance(examp_seq.feature, ClassLabel):
                    names = cast(ClassLabel, examp_seq.feature).names
                    new_example[examp_k] = [cls._replace_one(names, x) for x in examp_v]
                # Sequence of anything else
                else:
                    new_example[examp_k] = examp_v
            # Anything else
            else:
                new_example[examp_k] = examp_v
        return new_example

    def load_raw(
        self, data: str | DatalabLoaderOption, source: Source
    ) -> FileLoaderReturn:
        """See FileLoader.load_raw."""
        config = narrow(DatalabLoaderOption, data)
        serializer = PrimitiveSerializer()

        # load customized features from global config files
        customized_features_from_config = get_customized_features()
        if config.dataset in customized_features_from_config:
            ds_feats = customized_features_from_config[config.dataset]
            for level_name, level_feats in ds_feats["custom_features"].items():
                if level_name != "example":
                    raise NotImplementedError(
                        "currently custom features are only "
                        "supported on the example level, but got "
                        f"{level_name}"
                    )
                parsed_level_feats = {
                    # See https://github.com/python/mypy/issues/4717
                    k: narrow(FeatureType, serializer.deserialize(v))  # type: ignore
                    for k, v in level_feats.items()
                }
                new_features = config.custom_features.get(level_name, {})
                new_features.update(parsed_level_feats)
                config.custom_features[level_name] = new_features
            config.custom_analyses.extend(
                [
                    narrow(Analysis, serializer.deserialize(analysis))  # type: ignore
                    for analysis in ds_feats["custom_analyses"]
                ]
            )

        dataset = load_dataset(
            config.dataset, config.subdataset, split=config.split, streaming=False
        )
        # TODO(gneubig): patch for an inconsistency in datalab, where DatasetDict
        #  doesn't have info
        if isinstance(dataset, DatasetDict) or isinstance(dataset, IterableDatasetDict):
            raise ValueError("Cannot handle DatasetDict returns")
        info = dataset.info

        # update src_name based on task schema (e.g., text1_column => sentence1)
        ignore_tasks = ["machine-translation", "code-generation"]
        if info.task_templates[0].task not in ignore_tasks:
            for idx in range(len(self._fields)):
                src_name = cast(str, self._fields[idx].src_name)
                self._fields[idx].src_name = getattr(info.task_templates[0], src_name)

        # Infer metadata from the dataset
        metadata = FileLoaderMetadata(
            custom_features=config.custom_features,
            custom_analyses=config.custom_analyses,
        )

        if info.languages is not None:
            metadata.supported_languages = info.languages
            # Infer languages:
            # If only one language is supported, set both source and target to that
            # language. If two are supported set source to lang[0], target to lang[1].
            # Otherwise, do not infer the language at all.
            if (
                metadata.supported_languages
                and len(metadata.supported_languages) > 0
                and len(metadata.supported_languages) < 3
            ):
                metadata.source_language = metadata.supported_languages[0]
                metadata.target_language = metadata.supported_languages[
                    0 if len(metadata.supported_languages) == 1 else 1
                ]
            if info.task_templates is not None:
                tt = info.task_templates
                metadata.supported_tasks = list(
                    itertools.chain.from_iterable(
                        [[x.task] + x.task_categories for x in tt]
                    )
                )
        # Return
        return FileLoaderReturn(
            [self._replace_labels(info.features, x) for x in dataset],
            metadata=metadata,
        )


class TextFileLoader(FileLoader):
    """Loads a text file where each line is a different sample.

    Only one field is allowed. It is often used for predicted outputs of text generation
    models.
    """

    def __init__(
        self,
        target_name: str = "output",
        dtype: DType = str,
        strip_before_parsing: Optional[bool] = None,
    ) -> None:
        """Constructor.

        src_name is not used for this file loader, it overrides the base load method.

        Args:
            target_name: The name of the target field.
            dtype: The type of the field.
            strip_before_parsing: Whether to strip white space before parsing.
        """
        super().__init__(
            [FileLoaderField("_", target_name, dtype, strip_before_parsing)],
            use_idx_as_id=True,
        )

    @classmethod
    def load_raw(
        cls, data: str | DatalabLoaderOption, source: Source
    ) -> FileLoaderReturn:
        """See FileLoader.load_raw."""
        data = narrow(str, data)
        if source == Source.in_memory:
            return FileLoaderReturn(data.splitlines())
        elif source == Source.local_filesystem:
            with open(data, "r", encoding="utf8") as f:
                return FileLoaderReturn(f.readlines())
        raise NotImplementedError

    def validate(self) -> None:
        """See FileLoader.validate."""
        super().validate()
        if len(self._fields) != 1:
            raise ValueError("Text File Loader only takes one field")

    def load(
        self,
        data: str | DatalabLoaderOption,
        source: Source,
        field_mapping: dict[str, str] | None = None,
    ) -> FileLoaderReturn:
        """See FileLoader.load."""
        raw_data = self.load_raw(data, source)
        data_list: list[str] = raw_data.samples
        parsed_data_points: list[dict] = []

        for idx, data_point in enumerate(data_list):
            parsed_data_point = {}
            field = self._fields[0]
            parsed_data_point[field.target_name] = self.parse_data(data_point, field)
            self.generate_id(parsed_data_point, idx)
            parsed_data_points.append(parsed_data_point)
        return FileLoaderReturn(parsed_data_points)
