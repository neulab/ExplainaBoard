from explainaboard import feature
from explainaboard.processors.processor_registry import register_processor
from explainaboard.tasks import TaskType
from .conditional_generation import ConditionalGenerationProcessor


@register_processor(TaskType.machine_translation)
class MachineTranslationProcessor(ConditionalGenerationProcessor):
    _task_type = TaskType.machine_translation
    _default_metrics = ["bleu"]

    def __init__(self):
        super().__init__()
        # Inherit features from parent class and add new child-specific features
        _features = super()._features.update(  # noqa
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

    def _get_attr_compression(self, existing_features: dict):
        return len(existing_features["source"].split(" ")) / len(
            existing_features["reference"].split(" ")
        )
