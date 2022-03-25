import copy

from explainaboard import feature
from explainaboard.processors.processor_registry import register_processor
from explainaboard.tasks import TaskType
from .conditional_generation import ConditionalGenerationProcessor


@register_processor(TaskType.machine_translation)
class MachineTranslationProcessor(ConditionalGenerationProcessor):
    _task_type = TaskType.machine_translation
    _default_metrics = ["bleu"]
    _features = None

    def __init__(self):
        # Inherit features from parent class and add new child-specific features
        if MachineTranslationProcessor._features is None:
            MachineTranslationProcessor._features = copy.deepcopy(
                ConditionalGenerationProcessor._features
            )
            MachineTranslationProcessor._features.update(
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
        super().__init__()

    def _get_attr_compression(self, existing_features: dict):
        return len(self._tokenizer(existing_features["source"])) / len(
            self._tokenizer(existing_features["reference"])
        )
