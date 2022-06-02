from __future__ import annotations

from explainaboard import feature, TaskType
from explainaboard.info import SysOutputInfo
from explainaboard.metric import MetricConfig, SeqCorrectCountConfig
from explainaboard.processors.processor import Processor
from explainaboard.processors.processor_registry import register_processor
from explainaboard.utils.typing_utils import unwrap


@register_processor(TaskType.grammatical_error_correction)
class GrammaticalErrorCorrection(Processor):
    @classmethod
    def task_type(cls) -> TaskType:
        return TaskType.grammatical_error_correction

    @classmethod
    def default_features(cls) -> feature.Features:
        return feature.Features(
            {
                "text": feature.Value("string"),
                "edits": feature.Dict(
                    feature={
                        "start_idx": feature.Sequence(feature=feature.Value("int32")),
                        "end_idx": feature.Sequence(feature=feature.Value("int32")),
                        "corrections": feature.Sequence(
                            feature=feature.Sequence(feature=feature.Value("string"))
                        ),
                    }
                ),
                "text_length": feature.Value(
                    dtype="float",
                    description="length of the text",
                    is_bucket=True,
                    bucket_info=feature.BucketInfo(
                        method="bucket_attribute_specified_bucket_value",
                        number=4,
                        setting=(),
                    ),
                ),
            }
        )

    @classmethod
    def default_metrics(
        cls, source_language=None, target_language=None
    ) -> list[MetricConfig]:
        return [SeqCorrectCountConfig(name='SeqCorrectCount')]

    # --- Feature functions accessible by ExplainaboardBuilder._get_feature_func()
    def _get_text_length(self, sys_info: SysOutputInfo, existing_features: dict):
        return len(unwrap(sys_info.source_tokenizer)(existing_features["text"]))

    def _get_true_label(self, data_point):
        """
        Get the true label from a data point. Overloaded from parent class.
        :param data_point: the data point under consideration
        :return: the true label for the output
        """

        return data_point["edits"]

    def _get_predicted_label(self, data_point):
        """
        Get the predicted label from a data point. Overloaded from parent class.
        :param data_point: the data point under consideration
        :return: the predicted label for the output
        """
        return data_point["predicted_edits"]
