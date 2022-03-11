from typing import Iterable, Optional
from explainaboard.info import SysOutputInfo, Performance, BucketPerformance, Table
import copy
from eaas import Config, Client
from datalabs import load_dataset
from explainaboard.utils.analysis import *  # noqa
from explainaboard.utils.db_api import *

class ExplainaboardBuilder:

    def __init__(
        self,
        info: SysOutputInfo,
        system_output_object: Iterable[dict],
        feature_table: Optional[Table] = {},
        user_defined_feature_config = None,
    ):
        self._info = copy.deepcopy(info)
        self._system_output: Iterable[dict] = system_output_object
        self._user_defined_feature_config = user_defined_feature_config
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

    def _init_statistics(self, get_statistics):
        self.statistics = None
        if None != self._info.dataset_name:
            dataset_name = self._info.dataset_name
            split_name = "train"
            sub_dataset = None if self._info.sub_dataset_name == "default" else self._info.sub_dataset_name
            try:
                # read statistics from db
                response = read_statistics_from_db(dataset_name, sub_dataset)
                message = json.loads(response.text.replace("null", ""))["message"]
                eprint(message)
                if message == "success" and self._info.reload_stat == True:
                    self.statistics = json.loads(response.content)['content']
                elif message == "success" and self._info.reload_stat == False or message == "the dataset does not include the information of _stat":
                    dataset = load_dataset(self._info.dataset_name, self._info.sub_dataset_name)
                    if len(dataset[split_name]._stat) == 0 or self._info.reload_stat == False:  # calculate the statistics (_stat) when _stat is {} or `reload_stat` is False
                        new_train = dataset[split_name].apply(get_statistics, mode="local")

                        self.statistics = new_train._stat
                        # self.statistics = dataset['train']._stat
                        # write statistics to db
                        eprint("saving to database")
                        response = write_statistics_from_db(dataset_name, sub_dataset, content=self.statistics)
                        eprint(response.content)
                else:  # dataset does not exist
                    eprint(
                        "The dataset hasn't been supported by DataLab so no training set dependent features will be supported by ExplainaBoard."
                        "You can add the dataset by: https://github.com/ExpressAI/DataLab/blob/main/docs/SDK/add_new_datasets_into_sdk.md")
            except FileNotFoundError as err:
                eprint(
                    "The dataset hasn't been supported by DataLab so no training set dependent features will be supported by ExplainaBoard."
                    "You can add the dataset by: https://github.com/ExpressAI/DataLab/blob/main/docs/SDK/add_new_datasets_into_sdk.md")



    def _get_feature_func(self, func_name):
        return getattr(self, f'_get_{func_name}')

    def _get_eaas_client(self):
        if not self._eaas_client:
            self._eaas_config = Config()
            self._eaas_client = Client()
            self._eaas_client.load_config(self._eaas_config)  # The config you have created above
        return self._eaas_client
