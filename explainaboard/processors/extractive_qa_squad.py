from typing import Iterable
from explainaboard import feature
from explainaboard.tasks import TaskType
from explainaboard.processors.processor import Processor
from explainaboard.processors.processor_registry import register_processor
from explainaboard.builders.extractive_qa import QASquadExplainaboardBuilder

@register_processor(TaskType.extractive_qa_squad)
class QASqudProcessor(Processor):
    _task_type = TaskType.extractive_qa_squad
    _features = feature.Features({
        "title": feature.Value("string"),
        "context": feature.Value("string"),
        "question": feature.Value("string"),
        "id": feature.Value("string"),
        "true_answers": feature.Sequence(feature.Value("string")),
        "predicted_answer":feature.Value("string"),
        "context_length": feature.Value(dtype="float",
                                        description="context length",
                                         is_bucket=True,
                                         bucket_info=feature.BucketInfo(
                                             _method="bucket_attribute_specified_bucket_value",
                                             _number=4,
                                             _setting=())),
        "question_length": feature.Value(dtype="float",
                                         description="question length",
                                      is_bucket=True,
                                      bucket_info=feature.BucketInfo(
                                          _method="bucket_attribute_specified_bucket_value",
                                          _number=4,
                                          _setting=())),
        "answer_length": feature.Value(dtype="float",
                                       description="answer length",
                                         is_bucket=True,
                                         bucket_info=feature.BucketInfo(
                                             _method="bucket_attribute_specified_bucket_value",
                                             _number=4,
                                             _setting=())),
        # "sim_context_question": feature.Value(dtype="float",
        #                                is_bucket=True,
        #                                bucket_info=feature.BucketInfo(
        #                                    _method="bucket_attribute_specified_bucket_value",
        #                                    _number=4,
        #                                    _setting=()))
    })

    def __init__(self, metadata: dict, system_output_data: Iterable[dict]) -> None:
        if metadata == None:
            metadata = {}
        if "task_name" not in metadata.keys():
            metadata["task_name"] = TaskType.extractive_qa_squad.value
        if "metric_names" not in metadata.keys():
            metadata["metric_names"] = ["f1_score_qa","exact_match_qa"]
        super().__init__(metadata, system_output_data)
        self._builder = QASquadExplainaboardBuilder(self._system_output_info, system_output_data)