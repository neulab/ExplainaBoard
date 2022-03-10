from typing import Iterable, Optional
from explainaboard.info import SysOutputInfo, Performance, BucketPerformance, Table
import copy
from eaas import Config, Client


class ExplainaboardBuilder:

    def __init__(
        self,
        info: SysOutputInfo,
        system_output_object: Iterable[dict],
        feature_table: Optional[Table] = {},
        user_defined_feature_configs = None,
        gen_kwargs: dict = None,
    ):
        self._info = copy.deepcopy(info)
        self._system_output: Iterable[dict] = system_output_object
        self._user_defined_feature_configs = user_defined_feature_configs
        self.gen_kwargs = gen_kwargs
        self._data: Table = feature_table
        # _samples_over_bucket_true: Dict(feature_name, bucket_name, sample_id_true_label):
        # samples in different buckets
        self._samples_over_bucket = {}
        # _performances_over_bucket: performance in different bucket: Dict(feature_name, bucket_name, performance)
        self._performances_over_bucket = {}
        self.score_dic = None

        # Things to use only if necessary
        self._eaas_config = None
        self._eaas_client = None

    def _get_eaas_client(self):
        if not self._eaas_client:
            self._eaas_config = Config()
            self._eaas_client = Client()
            self._eaas_client.load_config(self._eaas_config)  # The config you have created above
        return self._eaas_client
