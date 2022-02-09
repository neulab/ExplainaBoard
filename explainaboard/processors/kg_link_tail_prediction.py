from typing import Iterable
from explainaboard import feature
from explainaboard.tasks import TaskType
from explainaboard.processors.processor import Processor
from explainaboard.processors.processor_registry import register_processor
from explainaboard.builders.kg_link_tail_prediction import KGLTPExplainaboardBuilder


@register_processor(TaskType.kg_link_tail_prediction)
class KGLinkTailPredictionProcessor(Processor):
    _task_type = TaskType.kg_link_tail_prediction
    _features = feature.Features({
        "true_head": feature.Value("string"),
        "link": feature.Value("string"),
        "true_tail": feature.Value("string"),
        "predicted_tails": feature.Sequence(feature.Value("string")),
        # ============================================
        # START WARM-UP TASKS
        "tail_entity_length": feature.Value(
            dtype = "float",
            description = "number of words in the tail entity",
            is_bucket = True,
            bucket_info = feature.BucketInfo(
                _method = "bucket_attribute_specified_bucket_value",
                _number = 4,
                _setting = ()
            )
        ),
        "head_entity_length": feature.Value(
            dtype="float",
            description="number of words in the head entity",
            is_bucket=True,
            bucket_info=feature.BucketInfo(
                _method="bucket_attribute_specified_bucket_value",
                _number=4,
                _setting=()
            )
        ),
        "tail_fre": feature.Value(
            dtype="float",
            description="the frequency of tail entity in the training set",
            is_bucket=True,
            bucket_info=feature.BucketInfo(
                _method="bucket_attribute_specified_bucket_value",
                _number=4,
                _setting=()
            )
        ),
        "link_fre": feature.Value(
            dtype="float",
            description="the frequency of link relation in the training set",
            is_bucket=True,
            bucket_info=feature.BucketInfo(
                _method="bucket_attribute_specified_bucket_value",
                _number=4,
                _setting=()
            )
        ),
        "head_fre": feature.Value(
            dtype="float",
            description="the frequency of head relation in the training set",
            is_bucket=True,
            bucket_info=feature.BucketInfo(
                _method="bucket_attribute_specified_bucket_value",
                _number=4,
                _setting=()
            )
        ),

    })


    def __init__(self, metadata: dict, system_output_data: Iterable[dict]) -> None:
        if metadata == None:
            metadata = {}
        if "task_name" not in metadata.keys():
            metadata["task_name"] = TaskType.kg_link_tail_prediction.value
        if "metric_names" not in metadata.keys():
            metadata["metric_names"] = ["Hits"]
        super().__init__(metadata, system_output_data)
        self._builder = KGLTPExplainaboardBuilder(self._system_output_info, system_output_data)
