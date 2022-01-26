from typing import Iterable
from explainaboard import feature
from explainaboard.tasks import TaskType
from explainaboard.processors.processor import Processor
from explainaboard.processors.processor_registry import register_processor
from explainaboard.builders.hellaswag import HellaswagExplainaboardBuilder

@register_processor(TaskType.hellaswag)
class HellaswagProcessor(Processor):
    _task_type = TaskType.hellaswag
    _features = feature.Features({
        "ctx_a": feature.Value("string", is_bucket=False),
        "ctx_b": feature.Value("string", is_bucket=False),
        "ctx": feature.Value("string", is_bucket=False),
        "endings": feature.Sequence(feature.Value("string")),
        "true_label": feature.ClassLabel(names=[
                                                "1",
                                                "0"
                                                ], is_bucket=False),
        "predicted_label": feature.ClassLabel(names=[
                                                     "1",
                                                     "0"
                                                     ], is_bucket=False),
        "ind": feature.Value(dtype="float",
                                         is_bucket=True,
                                         bucket_info=feature.BucketInfo(
                                             _method="bucket_attribute_specified_bucket_value",
                                             _number=4,
                                             _setting=())),
        "activity_label": feature.Value("string", is_bucket=True,
                                        bucket_info=feature.BucketInfo(
                                            _method="bucket_attribute_discrete_value",
                                            _number=10,
                                            _setting=1),
                                        ),
        "ctx_length": feature.Value(dtype="float",
                                         is_bucket=True,
                                         bucket_info=feature.BucketInfo(
                                             _method="bucket_attribute_specified_bucket_value",
                                             _number=4,
                                             _setting=())),
        "ctx_a_length_divided_b": feature.Value(dtype="float",
                                    is_bucket=True,
                                    bucket_info=feature.BucketInfo(
                                        _method="bucket_attribute_specified_bucket_value",
                                        _number=4,
                                        _setting=())),
        "true_answer_length": feature.Value(dtype="float",
                                    is_bucket=True,
                                    bucket_info=feature.BucketInfo(
                                        _method="bucket_attribute_specified_bucket_value",
                                        _number=4,
                                        _setting=())),
        "similarity_ctx_true_answer": feature.Value(dtype="float",
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
        self._builder = HellaswagExplainaboardBuilder(self._system_output_info, system_output_data) # easy to make mistake