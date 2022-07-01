from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class BucketAnalysis:
    """
    The class is used to define a dataclass for bucketing strategy
    Args:
        feature: the name of the feature to bucket
        method: the bucket strategy
        number: the number of buckets to be bucketed
        setting: hyper-paraterms of bucketing
    """

    feature: str
    method: str = "bucket_attribute_specified_bucket_value"
    number: int = 4
    setting: Any = 1  # For different bucket_methods, the settings are diverse
    _type: Optional[str] = None

    def __post_init__(self):
        self._type: str = self.__class__.__name__