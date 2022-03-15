from typing import List
from explainaboard.info import Result, SysOutputInfo
from explainaboard import feature
from explainaboard.tasks import TaskType
from explainaboard.processors.processor import Processor
from explainaboard.processors.processor_registry import register_processor
from explainaboard.builders.hellaswag import HellaswagExplainaboardBuilder


@register_processor(TaskType.hellaswag)
class HellaswagProcessor(Processor):
    _task_type = TaskType.hellaswag
    _features = feature.Features(
        {
            "ctx_a": feature.Value("string", is_bucket=False),
            "ctx_b": feature.Value("string", is_bucket=False),
            "ctx": feature.Value("string", is_bucket=False),
            "endings": feature.Sequence(feature.Value("string")),
            "true_label": feature.ClassLabel(names=["1", "0"], is_bucket=False),
            "predicted_label": feature.ClassLabel(names=["1", "0"], is_bucket=False),
            "ind": feature.Value(
                dtype="float",
                is_bucket=True,
                bucket_info=feature.BucketInfo(
                    method="bucket_attribute_specified_bucket_value",
                    number=4,
                    setting=(),
                ),
            ),
            "activity_label": feature.Value(
                "string",
                is_bucket=True,
                bucket_info=feature.BucketInfo(
                    method="bucket_attribute_discrete_value", number=10, setting=1
                ),
            ),
            "ctx_length": feature.Value(
                dtype="float",
                is_bucket=True,
                bucket_info=feature.BucketInfo(
                    method="bucket_attribute_specified_bucket_value",
                    number=4,
                    setting=(),
                ),
            ),
            "ctx_a_length_divided_b": feature.Value(
                dtype="float",
                is_bucket=True,
                bucket_info=feature.BucketInfo(
                    method="bucket_attribute_specified_bucket_value",
                    number=4,
                    setting=(),
                ),
            ),
            "true_answer_length": feature.Value(
                dtype="float",
                is_bucket=True,
                bucket_info=feature.BucketInfo(
                    method="bucket_attribute_specified_bucket_value",
                    number=4,
                    setting=(),
                ),
            ),
            "similarity_ctx_true_answer": feature.Value(
                dtype="float",
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
            metadata["task_name"] = TaskType.text_classification.value
        if "metric_names" not in metadata.keys():
            metadata["metric_names"] = ["Accuracy"]
        sys_info = SysOutputInfo.from_dict(metadata)
        sys_info.features = self._features
        builder = HellaswagExplainaboardBuilder()
        return builder.run(sys_info, sys_output)
