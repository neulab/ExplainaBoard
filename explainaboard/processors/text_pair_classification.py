from typing import List
from explainaboard.info import Result, SysOutputInfo
from explainaboard import feature
from explainaboard.tasks import TaskType
from explainaboard.processors.processor import Processor
from explainaboard.processors.processor_registry import register_processor
from explainaboard.builders.text_pair_classification import (
    TextPairClassificationExplainaboardBuilder,
)


@register_processor(TaskType.text_pair_classification)
class TextPairClassificationProcessor(Processor):
    _task_type = TaskType.text_classification
    _features = feature.Features(
        {
            "text": feature.Value("string"),
            "true_label": feature.ClassLabel(names=["1", "0"], is_bucket=False),
            "predicted_label": feature.ClassLabel(names=["1", "0"], is_bucket=False),
            "label": feature.Value(
                dtype="string",
                is_bucket=True,
                bucket_info=feature.BucketInfo(
                    method="bucket_attribute_discrete_value", number=4, setting=1
                ),
            ),
            "text1_length": feature.Value(
                dtype="float",
                description="text1 length",
                is_bucket=True,
                bucket_info=feature.BucketInfo(
                    method="bucket_attribute_specified_bucket_value",
                    number=4,
                    setting=(),
                ),
            ),
            "text2_length": feature.Value(
                dtype="float",
                description="text2 length",
                is_bucket=True,
                bucket_info=feature.BucketInfo(
                    method="bucket_attribute_specified_bucket_value",
                    number=4,
                    setting=(),
                ),
            ),
            "similarity": feature.Value(
                dtype="float",
                description="two texts' similarity",
                is_bucket=True,
                bucket_info=feature.BucketInfo(
                    method="bucket_attribute_specified_bucket_value",
                    number=4,
                    setting=(),
                ),
            ),
            "text1_divided_text2": feature.Value(
                dtype="float",
                description="diff of two texts' length",
                is_bucket=True,
                bucket_info=feature.BucketInfo(
                    method="bucket_attribute_specified_bucket_value",
                    number=4,
                    setting=(),
                ),
            ),
            "num_oov": feature.Value(
                dtype="float",
                description="the number of out-of-vocabulary words",
                is_bucket=True,
                bucket_info=feature.BucketInfo(
                    method="bucket_attribute_specified_bucket_value",
                    number=4,
                    setting=(),
                ),
                require_training_set=True,
            ),
            "fre_rank": feature.Value(
                dtype="float",
                description="the average rank of each work based on its frequency in training set",
                is_bucket=True,
                bucket_info=feature.BucketInfo(
                    method="bucket_attribute_specified_bucket_value",
                    number=4,
                    setting=(),
                ),
                require_training_set=True,
            ),
        }
    )

    def __init__(self) -> None:
        super().__init__()

    def process(self,
                metadata: dict,
                sys_output: List[dict]) -> Result:
        if metadata is None:
            metadata = {}
        if "task_name" not in metadata.keys():
            metadata["task_name"] = TaskType.qa_multiple_choice.value
        if "metric_names" not in metadata.keys():
            metadata["metric_names"] = ["Accuracy"]
        sys_info = SysOutputInfo.from_dict(metadata)
        sys_info.features = self._features
        builder = TextPairClassificationExplainaboardBuilder()
        return builder.run(sys_info, sys_output)
