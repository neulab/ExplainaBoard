from typing import Iterable
from explainaboard import feature
from explainaboard.tasks import TaskType
from explainaboard.processors.processor import Processor
from explainaboard.processors.processor_registry import register_processor
from explainaboard.builders.summarization import SummExplainaboardBuilder

@register_processor(TaskType.summarization)
class TextSummarizationProcessor(Processor):
    _task_type = TaskType.summarization
    _features = feature.Features({
        "source": feature.Value("string"),
        "reference": feature.Value("string"),
        "hypothesis": feature.Value("string"),
        "attr_source_len": feature.Value(dtype="float",
                                         description="length of source document",
                                         is_bucket=True,
                                         bucket_info=feature.BucketInfo(
                                             _method="bucket_attribute_specified_bucket_value",
                                             _number=4,
                                             _setting=())),
        "attr_compression": feature.Value(dtype="float",
                                          description="compression",
                                         is_bucket=True,
                                         bucket_info=feature.BucketInfo(
                                             _method="bucket_attribute_specified_bucket_value",
                                             _number=4,
                                             _setting=())),
        "attr_copy_len": feature.Value(dtype="float",
                                       description="copy length",
                                          is_bucket=True,
                                          bucket_info=feature.BucketInfo(
                                              _method="bucket_attribute_specified_bucket_value",
                                              _number=4,
                                              _setting=())),
        "attr_coverage": feature.Value(dtype="float",
                                       description="coverage",
                                       is_bucket=True,
                                       bucket_info=feature.BucketInfo(
                                           _method="bucket_attribute_specified_bucket_value",
                                           _number=4,
                                           _setting=())),
        "attr_novelty": feature.Value(dtype="float",
                                      description="novelty",
                                       is_bucket=True,
                                       bucket_info=feature.BucketInfo(
                                           _method="bucket_attribute_specified_bucket_value",
                                           _number=4,
                                           _setting=()))

    })


    def __init__(self, metadata: dict, system_output_data: Iterable[dict]) -> None:
        if metadata == None:
            metadata = {}
        if "task_name" not in metadata.keys():
            metadata["task_name"] = TaskType.summarization.value
        if "metric_names" not in metadata.keys():
            #metadata["metric_names"] = ["chrf","bart_score_summ","bleu","comet","mover_score","prism"]
            metadata["metric_names"] = ["bleu"]

        super().__init__(metadata, system_output_data)
        self._builder = SummExplainaboardBuilder(self._system_output_info, system_output_data)