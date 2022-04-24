from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache

from datalabs import aggregating
from datalabs.operations.featurize.plugins.summarization.sum_attribute import (
    SUMAttribute,
)
from datalabs.operations.featurize.summarization import get_oracle_summary
import numpy

from explainaboard import feature, TaskType
from explainaboard.info import SysOutputInfo
from explainaboard.metric import EaaSMetricConfig, MetricConfig
from explainaboard.processors.conditional_generation import (
    ConditionalGenerationProcessor,
)
from explainaboard.processors.processor_registry import register_processor
import explainaboard.utils.feature_funcs
from explainaboard.utils.py_utils import hash_dict

# to calculate advanced features
from explainaboard.utils.tokenizer import Tokenizer

summary_attribute = SUMAttribute()


@hash_dict
@lru_cache(maxsize=10)
def get_oracle(existing_features: dict):
    """
    oracle_info =
        {
        "source":src,
        "reference":ref,
        "oracle_summary":oracle,
        "oracle_labels":labels,
        "oracle_score":max_score
        }
    """

    sample = {
        "text": existing_features["source"],
        "summary": existing_features["reference"],
    }
    oracle_info = get_oracle_summary.func(sample)

    index_of_oracles = [i for i, e in enumerate(oracle_info["oracle_labels"]) if e != 0]
    oracle_position = numpy.mean(index_of_oracles)

    return {
        "oracle_position": oracle_position,
        "oracle_score": oracle_info["oracle_score"],
    }


@register_processor(TaskType.summarization)
class SummarizationProcessor(ConditionalGenerationProcessor):
    @classmethod
    def task_type(cls) -> TaskType:
        return TaskType.summarization

    @classmethod
    def default_features(cls) -> feature.Features:
        f = super().default_features()
        f.update(
            feature.Features(
                {
                    "attr_compression": feature.Value(
                        dtype="float",
                        description="compression",
                        is_bucket=True,
                        bucket_info=feature.BucketInfo(
                            method="bucket_attribute_specified_bucket_value",
                            number=4,
                            setting=(),
                        ),
                    ),
                    "attr_copy_len": feature.Value(
                        dtype="float",
                        description="copy length",
                        is_bucket=True,
                        bucket_info=feature.BucketInfo(
                            method="bucket_attribute_specified_bucket_value",
                            number=4,
                            setting=(),
                        ),
                    ),
                    "attr_coverage": feature.Value(
                        dtype="float",
                        description="coverage",
                        is_bucket=True,
                        bucket_info=feature.BucketInfo(
                            method="bucket_attribute_specified_bucket_value",
                            number=4,
                            setting=(),
                        ),
                    ),
                    "attr_novelty": feature.Value(
                        dtype="float",
                        description="novelty",
                        is_bucket=True,
                        bucket_info=feature.BucketInfo(
                            method="bucket_attribute_specified_bucket_value",
                            number=4,
                            setting=(),
                        ),
                    ),
                    "oracle_score": feature.Value(
                        dtype="float",
                        description="the sample-level oracle score",
                        is_bucket=True,
                        bucket_info=feature.BucketInfo(
                            method="bucket_attribute_specified_bucket_value",
                            number=4,
                            setting=(),
                        ),
                    ),
                    "oracle_position": feature.Value(
                        dtype="float",
                        description="the sample-level oracle position",
                        is_bucket=True,
                        bucket_info=feature.BucketInfo(
                            method="bucket_attribute_specified_bucket_value",
                            number=4,
                            setting=(),
                        ),
                    ),
                }
            )
        )
        return f

    @classmethod
    def default_metrics(cls, language=None) -> list[MetricConfig]:
        return [
            EaaSMetricConfig(name='rouge1', language=language),
            EaaSMetricConfig(name='rouge2', language=language),
            EaaSMetricConfig(name='rougeL', language=language),
        ]

    def _get_oracle_position(self, sys_info: SysOutputInfo, existing_features: dict):
        return get_oracle(existing_features)["oracle_position"]

    def _get_oracle_score(self, sys_info: SysOutputInfo, existing_features: dict):
        return get_oracle(existing_features)["oracle_score"]

    def _get_attr_compression(self, sys_info: SysOutputInfo, existing_features: dict):
        res = summary_attribute.cal_attributes_each(
            existing_features["source"], existing_features["reference"]
        )
        return res["attr_compression"]

    def _get_attr_copy_len(self, sys_info: SysOutputInfo, existing_features: dict):
        res = summary_attribute.cal_attributes_each(
            existing_features["source"], existing_features["reference"]
        )
        return res["attr_copy_len"]

    def _get_attr_coverage(self, sys_info: SysOutputInfo, existing_features: dict):
        res = summary_attribute.cal_attributes_each(
            existing_features["source"], existing_features["reference"]
        )
        return res["attr_coverage"]

    def _get_attr_novelty(self, sys_info: SysOutputInfo, existing_features: dict):
        res = summary_attribute.cal_attributes_each(
            existing_features["source"], existing_features["reference"]
        )
        return res["attr_novelty"]

    @aggregating()
    def _statistics_func(self, samples: Iterator, tokenizer: Tokenizer):
        return explainaboard.utils.feature_funcs.accumulate_vocab_from_samples(
            samples, lambda x: x['summary'], tokenizer
        )
