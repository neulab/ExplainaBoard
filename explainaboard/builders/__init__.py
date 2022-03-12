from typing import Iterable, Optional
from explainaboard.info import SysOutputInfo, Table, Performance
from explainaboard.metric import Accuracy, F1score
import copy
from tqdm import tqdm
from eaas import Config, Client
from datalabs import load_dataset
from explainaboard.utils.analysis import *  # noqa
from explainaboard.utils.db_api import *


class ExplainaboardBuilder:
    def __init__(
        self,
        info: SysOutputInfo,
        system_output_object: Iterable[dict],
        feature_table: Optional[Table] = None,
        user_defined_feature_config=None,
    ):
        self._info = copy.deepcopy(info)
        self._system_output: Iterable[dict] = system_output_object
        self._user_defined_feature_config = user_defined_feature_config
        self._data: Table = feature_table if feature_table else {}
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
        if self._info.dataset_name is not None:
            dataset_name = self._info.dataset_name
            split_name = "train"
            sub_dataset = (
                None
                if self._info.sub_dataset_name == "default"
                else self._info.sub_dataset_name
            )
            try:
                # read statistics from db
                response = read_statistics_from_db(dataset_name, sub_dataset)
                message = json.loads(response.text.replace("null", ""))["message"]
                eprint(message)
                if message == "success" and self._info.reload_stat:
                    self.statistics = json.loads(response.content)['content']
                elif (
                    message == "success"
                    and not self._info.reload_stat
                    or message
                    == "the dataset does not include the information of _stat"
                ):
                    dataset = load_dataset(
                        self._info.dataset_name, self._info.sub_dataset_name
                    )
                    if (
                        len(dataset[split_name]._stat) == 0
                        or not self._info.reload_stat
                    ):  # calculate the statistics (_stat) when _stat is {} or `reload_stat` is False
                        new_train = dataset[split_name].apply(
                            get_statistics, mode="local"
                        )

                        self.statistics = new_train._stat
                        # self.statistics = dataset['train']._stat
                        # write statistics to db
                        eprint("saving to database")
                        response = write_statistics_from_db(
                            dataset_name, sub_dataset, content=self.statistics
                        )
                        eprint(response.content)
                else:  # dataset does not exist
                    eprint(
                        "The dataset hasn't been supported by DataLab so no training set dependent features will be supported by ExplainaBoard."
                        "You can add the dataset by: https://github.com/ExpressAI/DataLab/blob/main/docs/SDK/add_new_datasets_into_sdk.md"
                    )
            except FileNotFoundError as err:
                eprint(
                    "The dataset hasn't been supported by DataLab so no training set dependent features will be supported by ExplainaBoard."
                    "You can add the dataset by: https://github.com/ExpressAI/DataLab/blob/main/docs/SDK/add_new_datasets_into_sdk.md"
                )

    def _get_feature_func(self, func_name):
        return getattr(self, f'_get_{func_name}')

    def _get_eaas_client(self):
        if not self._eaas_client:
            self._eaas_config = Config()
            self._eaas_client = Client()
            self._eaas_client.load_config(
                self._eaas_config
            )  # The config you have created above
        return self._eaas_client

    def get_overall_performance(self):
        predicted_labels, true_labels = [], []

        for _id, feature_table in self._data.items():

            predicted_labels.append(feature_table["predicted_label"])
            true_labels.append(feature_table["true_label"])

        for metric_name in self._info.metric_names:
            # TODO(gneubig): we should try to get rid of these "eval()" calls, as they're dangerous
            one_metric = eval(metric_name)(
                true_labels=true_labels,
                predicted_labels=predicted_labels,
                is_print_confidence_interval=self._info.results.is_print_confidence_interval,
            )
            overall_value_json = one_metric.evaluate()

            overall_value = overall_value_json["value"]
            confidence_score_low = overall_value_json["confidence_score_low"]
            confidence_score_up = overall_value_json["confidence_score_up"]
            overall_performance = Performance(
                metric_name=metric_name,
                value=float(format(overall_value, '.4g')),
                confidence_score_low=float(format(confidence_score_low, '.4g')),
                confidence_score_up=float(format(confidence_score_up, '.4g')),
            )

            if self._info.results.overall is None:
                self._info.results.overall = {}
                self._info.results.overall[metric_name] = overall_performance
            else:
                self._info.results.overall[metric_name] = overall_performance

    def _bucketing_samples(self, sysout_iterator):

        feature_to_sample_address_to_value = {}

        # Preparation for bucketing
        for _id, dict_sysout in sysout_iterator:

            sample_address = str(_id)  # this could be abstracted later
            for feature_name in self._info.features.get_bucket_features():
                if feature_name not in feature_to_sample_address_to_value.keys():
                    feature_to_sample_address_to_value[feature_name] = {}
                else:
                    feature_to_sample_address_to_value[feature_name][
                        sample_address
                    ] = dict_sysout[feature_name]

        # Bucketing
        for feature_name in tqdm(
            self._info.features.get_bucket_features(), desc="bucketing"
        ):
            # print(f"Feature Name: {feature_name}\n"
            #       f"Bucket Hyper:\n function_name: {self._info.features[feature_name].bucket_info._method} \n"
            #       f"bucket_number: {self._info.features[feature_name].bucket_info._number}\n"
            #       f"bucket_setting: {self._info.features[feature_name].bucket_info._setting}\n")

            self._samples_over_bucket[feature_name] = eval(
                self._info.features[feature_name].bucket_info._method
            )(
                dict_obj=feature_to_sample_address_to_value[feature_name],
                bucket_number=self._info.features[feature_name].bucket_info._number,
                bucket_setting=self._info.features[feature_name].bucket_info._setting,
            )

            # print(f"self._samples_over_bucket.keys():\n{self._samples_over_bucket.keys()}")

            # evaluating bucket: get bucket performance
            self._performances_over_bucket[feature_name] = self.get_bucket_performance(
                feature_name
            )

    def _generate_report(self):
        dict_fine_grained = {}
        for feature_name, metadata in self._performances_over_bucket.items():
            dict_fine_grained[feature_name] = []
            for bucket_name, bucket_performance in metadata.items():
                bucket_name = beautify_interval(bucket_name)  # noqa

                # instantiation
                dict_fine_grained[feature_name].append(bucket_performance)

        self._info.results.fine_grained = dict_fine_grained

    def _print_bucket_info(self):
        for feature_name in self._performances_over_bucket.keys():
            print_dict(  # noqa
                self._performances_over_bucket[feature_name], feature_name
            )

    def run(self) -> SysOutputInfo:
        eb_generator = self._complete_feature()
        self._bucketing_samples(eb_generator)
        self.get_overall_performance()
        self._print_bucket_info()
        self._generate_report()
        return self._info
