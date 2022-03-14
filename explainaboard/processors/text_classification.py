from typing import Iterable, List
from explainaboard import feature
from explainaboard.info import SysOutputInfo, Result
from explainaboard.tasks import TaskType
from explainaboard.processors.processor import Processor
from explainaboard.processors.processor_registry import register_processor
from explainaboard.builders.text_classification import TCExplainaboardBuilder


@register_processor(TaskType.text_classification)
class TextClassificationProcessor(Processor):
    _task_type = TaskType.text_classification
    _features = feature.Features(
        {
            "text": feature.Value("string"),
            "true_label": feature.ClassLabel(names=["1", "0"], is_bucket=False),
            "predicted_label": feature.ClassLabel(names=["1", "0"], is_bucket=False),
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
                description="text length",
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
            "basic_words": feature.Value(
                dtype="float",
                description="the ratio of basic words",
                is_bucket=True,
                bucket_info=feature.BucketInfo(
                    method="bucket_attribute_specified_bucket_value",
                    number=4,
                    setting=(),
                ),
            ),
            "lexical_richness": feature.Value(
                dtype="float",
                description="lexical diversity",
                is_bucket=True,
                bucket_info=feature.BucketInfo(
                    method="bucket_attribute_specified_bucket_value",
                    number=4,
                    setting=(),
                ),
            ),
            "entity_number": feature.Value(
                dtype="float",
                description="the number of entities",
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
            "length_fre": feature.Value(
                dtype="float",
                description="the frequency of text length in training set",
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
            metadata["task_name"] = TaskType.text_classification.value
        if "metric_names" not in metadata.keys():
            metadata["metric_names"] = ["Accuracy"]
        sys_info = SysOutputInfo.from_dict(metadata)
        sys_info.features = self._features
        builder = TCExplainaboardBuilder()
        return builder.run(sys_info, sys_output)
