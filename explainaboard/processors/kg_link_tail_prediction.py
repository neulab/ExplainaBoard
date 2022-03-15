from typing import List
from explainaboard import feature
from explainaboard.tasks import TaskType
from explainaboard.info import SysOutputInfo
from explainaboard.processors.processor import Processor
from explainaboard.processors.processor_registry import register_processor
from explainaboard.builders.kg_link_tail_prediction import KGLTPExplainaboardBuilder


@register_processor(TaskType.kg_link_tail_prediction)
class KGLinkTailPredictionProcessor(Processor):
    _task_type = TaskType.kg_link_tail_prediction
    _features = feature.Features(
        {
            "true_head": feature.Value("string"),
            "link": feature.Value("string"),
            "true_tail": feature.Value("string"),
            "predicted_tails": feature.Sequence(feature.Value("string")),
            "tail_entity_length": feature.Value(
                dtype="float",
                description="number of words in the tail entity",
                is_bucket=True,
                bucket_info=feature.BucketInfo(
                    method="bucket_attribute_specified_bucket_value",
                    number=4,
                    setting=(),
                ),
            ),
            "head_entity_length": feature.Value(
                dtype="float",
                description="number of words in the head entity",
                is_bucket=True,
                bucket_info=feature.BucketInfo(
                    method="bucket_attribute_specified_bucket_value",
                    number=4,
                    setting=(),
                ),
            ),
            "tail_fre": feature.Value(
                dtype="float",
                description="the frequency of tail entity in the training set",
                is_bucket=True,
                bucket_info=feature.BucketInfo(
                    method="bucket_attribute_specified_bucket_value",
                    number=4,
                    setting=(),
                ),
                require_training_set=True,
            ),
            "link_fre": feature.Value(
                dtype="float",
                description="the frequency of link relation in the training set",
                is_bucket=True,
                bucket_info=feature.BucketInfo(
                    method="bucket_attribute_specified_bucket_value",
                    number=4,
                    setting=(),
                ),
                require_training_set=True,
            ),
            "head_fre": feature.Value(
                dtype="float",
                description="the frequency of head relation in the training set",
                is_bucket=True,
                bucket_info=feature.BucketInfo(
                    method="bucket_attribute_specified_bucket_value",
                    number=4,
                    setting=(),
                ),
                require_training_set=True,
            ),
            "symmetry": feature.Value(
                dtype="string",
                description="boolean feature: 'symmetric' or 'asymmetric'; more granularity to be added",
                is_bucket=True,
                bucket_info=feature.BucketInfo(
                    method="bucket_attribute_discrete_value", number=2, setting=1
                ),
            ),
            "entity_type_level": feature.Value(
                dtype="string",
                description="most specific (highest) entity type level of true tail entity",
                is_bucket=True,
                bucket_info=feature.BucketInfo(
                    method="bucket_attribute_discrete_value", number=8, setting=1
                ),
            ),
        }
    )
    _default_metrics = ["Hits", "MeanReciprocalRank"]

    def process(self, metadata: dict, sys_output: List[dict]) -> SysOutputInfo:
        self._builder = KGLTPExplainaboardBuilder(
            user_defined_feature_config=metadata.get(
                "user_defined_features_configs", None
            )
        )
        return super().process(metadata, sys_output)
