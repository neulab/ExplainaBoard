from typing import Any, List, Optional
from explainaboard import feature
from explainaboard.tasks import TaskType
from explainaboard.info import Result


class Processor:
    """Base case for task-based processor"""

    _features: feature.Features
    _task_type: TaskType

    def __init__(self) -> None:
        pass
        # self._metadata = {**metadata, "features": self._features}
        # self._system_output_info = SysOutputInfo.from_dict(self._metadata)
        # should really be a base type of builders
        # self._builder: Optional[Any] = None
        # # add user-defined features into features list
        # feature_configs = metadata.get("user_defined_features_configs", None)
        # if feature_configs is not None:
        #     for feature_name, feature_config in feature_configs.items():
        #         if feature_config["dtype"] == "string":
        #             self._features[feature_name] = feature.Value(
        #                 dtype="string",
        #                 description=feature_config["description"],
        #                 is_bucket=True,
        #                 bucket_info=feature.BucketInfo(
        #                     method="bucket_attribute_discrete_value",
        #                     number=feature_config["num_buckets"],
        #                     setting=1,
        #                 ),
        #             )
        #         elif feature_config['dtype'] == 'float':
        #             self._features[feature_name] = feature.Value(
        #                 dtype="float",
        #                 description=feature_config["description"],
        #                 is_bucket=True,
        #                 bucket_info=feature.BucketInfo(
        #                     method="bucket_attribute_specified_bucket_value",
        #                     number=feature_config["num_buckets"],
        #                     setting=(),
        #                 ),
        #             )
        #         else:
        #             raise NotImplementedError

    def process(self, metadata: dict, sys_output: List[dict]) -> Result:
        raise NotImplementedError
