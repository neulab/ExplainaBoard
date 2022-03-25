from typing import List

from explainaboard import feature
from explainaboard.processors.processor_registry import register_processor
from explainaboard.tasks import TaskType
from .conditional_generation import ConditionalGenerationProcessor


@register_processor(TaskType.machine_translation)
class MachineTranslationProcessor(ConditionalGenerationProcessor):
    @classmethod
    def task_type(cls) -> TaskType:
        return TaskType.machine_translation

    @classmethod
    def default_features(cls) -> feature.Features:
        f = super().default_features()
        f.update(
            feature.Features(
                {
                    # declaim task-specific features
                    "attr_compression": feature.Value(
                        dtype="float",
                        description="the ratio between source and reference length",
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
    def default_metrics(cls) -> List[str]:
        return ["bleu"]

    def _get_attr_compression(self, existing_features: dict):
        return len(self._tokenizer(existing_features["source"])) / len(
            self._tokenizer(existing_features["reference"])
        )
