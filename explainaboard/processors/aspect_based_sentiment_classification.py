from typing import List
from explainaboard.info import Result, SysOutputInfo
from explainaboard import feature
from explainaboard.tasks import TaskType
from explainaboard.processors.processor import Processor
from explainaboard.processors.processor_registry import register_processor
from explainaboard.builders.aspect_based_sentiment_classification import (
    ABSCExplainaboardBuilder,
)


@register_processor(TaskType.aspect_based_sentiment_classification)
class AspectBasedSentimentClassificationProcessor(Processor):
    _task_type = TaskType.aspect_based_sentiment_classification
    _features = feature.Features(
        {
            "aspect": feature.Value("string"),
            "text": feature.Value("string"),
            "true_label": feature.ClassLabel(
                names=["positive", "negative"], is_bucket=False
            ),
            "predicted_label": feature.ClassLabel(
                names=["positive", "negative"], is_bucket=False
            ),
            "label": feature.Value(
                dtype="string",
                description="category",
                is_bucket=True,
                bucket_info=feature.BucketInfo(
                    method="bucket_attribute_discrete_value", number=4, setting=1
                ),
            ),
            "sentence_length": feature.Value(
                dtype="float",
                description="sentence length",
                is_bucket=True,
                bucket_info=feature.BucketInfo(
                    method="bucket_attribute_specified_bucket_value",
                    number=4,
                    setting=(),
                ),
            ),
            "token_number": feature.Value(
                dtype="float",
                description="the number of chars",
                is_bucket=True,
                bucket_info=feature.BucketInfo(
                    method="bucket_attribute_specified_bucket_value",
                    number=4,
                    setting=(),
                ),
            ),
            "entity_number": feature.Value(
                dtype="float",
                description="entity numbers",
                is_bucket=True,
                bucket_info=feature.BucketInfo(
                    method="bucket_attribute_specified_bucket_value",
                    number=4,
                    setting=(),
                ),
            ),
            "aspect_length": feature.Value(
                dtype="float",
                description="aspect length",
                is_bucket=True,
                bucket_info=feature.BucketInfo(
                    method="bucket_attribute_specified_bucket_value",
                    number=4,
                    setting=(),
                ),
            ),
            "aspect_index": feature.Value(
                dtype="float",
                description="aspect position",
                is_bucket=True,
                bucket_info=feature.BucketInfo(
                    method="bucket_attribute_specified_bucket_value",
                    number=4,
                    setting=(),
                ),
            ),
        }
    )

    def __init__(self) -> None:
        super().__init__()

    def process(self, metadata: dict, sys_output: List[dict]) -> Result:
        if metadata is None:
            metadata = {}
        if "task_name" not in metadata.keys():
            metadata["task_name"] = TaskType.aspect_based_sentiment_classification.value
        if "metric_names" not in metadata.keys() or metadata["metric_names"] is None:
            metadata["metric_names"] = ["Accuracy"]
        sys_info = SysOutputInfo.from_dict(metadata)
        sys_info.features = self._features
        builder = ABSCExplainaboardBuilder()
        return builder.run(sys_info, sys_output)

        # explainaboard --task aspect-based-sentiment-classification --system_outputs ./data/system_outputs/absa/test-aspect.tsv
