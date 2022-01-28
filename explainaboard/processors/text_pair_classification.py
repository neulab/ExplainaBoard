from typing import Iterable
from explainaboard import feature
from explainaboard.tasks import TaskType
from explainaboard.processors.processor import Processor
from explainaboard.processors.processor_registry import register_processor
from explainaboard.builders.text_pair_classification import TextPairClassificationExplainaboardBuilder

@register_processor(TaskType.text_pair_classification)
class TextPairClassificationProcessor(Processor):
    _task_type = TaskType.text_classification
    _features = feature.Features({
        "text": feature.Value("string"),
        "true_label": feature.ClassLabel(names=[
                                                "1",
                                                "0"
                                                ], is_bucket=False),
        "predicted_label": feature.ClassLabel(names=[
                                                     "1",
                                                     "0"
                                                     ], is_bucket=False),
        "label": feature.Value(dtype="string",
                               is_bucket=True,
                               bucket_info=feature.BucketInfo(
                                   _method="bucket_attribute_discrete_value",
                                   _number=4,
                                   _setting=1)),
        "text1_length": feature.Value(dtype="float",
                                      description="text1 length",
                                         is_bucket=True,
                                         bucket_info=feature.BucketInfo(
                                             _method="bucket_attribute_specified_bucket_value",
                                             _number=4,
                                             _setting=())),
        "text2_length": feature.Value(dtype="float",
                                      description="text2 length",
                                      is_bucket=True,
                                      bucket_info=feature.BucketInfo(
                                          _method="bucket_attribute_specified_bucket_value",
                                          _number=4,
                                          _setting=())),
        "similarity": feature.Value(dtype="float",
                                    description="two texts' similarity",
                                      is_bucket=True,
                                      bucket_info=feature.BucketInfo(
                                          _method="bucket_attribute_specified_bucket_value",
                                          _number=4,
                                          _setting=())),
        "text1_divided_text2": feature.Value(dtype="float",
                                             description="diff of two texts' length",
                                    is_bucket=True,
                                    bucket_info=feature.BucketInfo(
                                        _method="bucket_attribute_specified_bucket_value",
                                        _number=4,
                                        _setting=())),

    })


    def __init__(self, metadata: dict, system_output_data: Iterable[dict]) -> None:
        if metadata == None:
            metadata = {}
        if "task_name" not in metadata.keys():
            metadata["task_name"] = TaskType.text_classification.value
        if "metric_names" not in metadata.keys():
            metadata["metric_names"] = ["Accuracy"]
        super().__init__(metadata, system_output_data)
        self._builder = TextPairClassificationExplainaboardBuilder(self._system_output_info, system_output_data)