from typing import Iterable
from explainaboard import feature
from explainaboard.tasks import TaskType
from explainaboard.processors.processor import Processor
from explainaboard.processors.processor_registry import register_processor
from explainaboard.builders.conditional_generation import CondGenExplainaboardBuilder


class ConditionalGenerationProcessor(Processor):
    _features = feature.Features(
        {
            "source": feature.Value("string"),
            "reference": feature.Value("string"),
            "hypothesis": feature.Value("string"),
            "attr_source_len": feature.Value(
                dtype="float",
                description="length of source document",
                is_bucket=True,
                bucket_info=feature.BucketInfo(
                    _method="bucket_attribute_specified_bucket_value",
                    _number=4,
                    _setting=(),
                ),
            ),
            "attr_compression": feature.Value(
                dtype="float",
                description="compression",
                is_bucket=True,
                bucket_info=feature.BucketInfo(
                    _method="bucket_attribute_specified_bucket_value",
                    _number=4,
                    _setting=(),
                ),
            ),
            "attr_copy_len": feature.Value(
                dtype="float",
                description="copy length",
                is_bucket=True,
                bucket_info=feature.BucketInfo(
                    _method="bucket_attribute_specified_bucket_value",
                    _number=4,
                    _setting=(),
                ),
            ),
            "attr_coverage": feature.Value(
                dtype="float",
                description="coverage",
                is_bucket=True,
                bucket_info=feature.BucketInfo(
                    _method="bucket_attribute_specified_bucket_value",
                    _number=4,
                    _setting=(),
                ),
            ),
            "attr_novelty": feature.Value(
                dtype="float",
                description="novelty",
                is_bucket=True,
                bucket_info=feature.BucketInfo(
                    _method="bucket_attribute_specified_bucket_value",
                    _number=4,
                    _setting=(),
                ),
            ),
        }
    )

    def __init__(self, metadata: dict, system_output_data: Iterable[dict]) -> None:
        if metadata is None:
            metadata = {}
        if "task_name" not in metadata.keys():
            metadata["task_name"] = TaskType.summarization.value
        if "metric_names" not in metadata.keys():
            # metadata["metric_names"] = ["chrf","bart_score_summ","bleu","comet","mover_score","prism"]
            metadata["metric_names"] = ["bleu"]

        super().__init__(metadata, system_output_data)
        self._builder = CondGenExplainaboardBuilder(
            self._system_output_info, system_output_data
        )


@register_processor(TaskType.summarization)
class SummarizationProcessor(ConditionalGenerationProcessor):
    _task_type = TaskType.summarization


@register_processor(TaskType.machine_translation)
class MachineTranslationProcessor(ConditionalGenerationProcessor):
    _task_type = TaskType.machine_translation
