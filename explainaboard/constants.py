from enum import Enum


class Source(str, Enum):
    in_memory = "in_memory"  # content has been loaded in memory
    local_filesystem = "local_filesystem"
    s3 = "s3"
    mongodb = "mongodb"


class FileType(str, Enum):
    json = "json"
    tsv = "tsv"
    csv = "csv"
    conll = "conll"  # for tagging task such as named entity recognition
