from typing import Iterable, Optional
from explainaboard.info import SysOutputInfo, Performance, BucketPerformance, Table
from explainaboard.utils.analysis import *  # noqa
from explainaboard.utils.eval_bucket import *  # noqa
import copy
import numpy
from tqdm import tqdm

from typing import Iterator, Dict, List
from datalabs import load_dataset
from datalabs.operations.aggregate.summarization import summarization_aggregating


from eaas import Config, Client

config = Config()
client = Client()
client.load_config(config)  # The config you have created above


@summarization_aggregating(name="get_statistics", contributor="datalab",
                                 task="summarization",
                                 description="Calculate the overall statistics (e.g., density) of a given summarization dataset")
def get_statistics(samples: Iterator):
    """
    Input:
    samples: [{
     "text":
     "summary":
    }]
    Output:dict:
    """

    vocab = {}
    length_fre = {}
    for sample in tqdm(samples):

        text, summary = sample["text"], sample["summary"]

        # Vocabulary info
        for w in (text + summary).split(" "):
            if w in vocab.keys():
                vocab[w] += 1
            else:
                vocab[w] = 1

    # the rank of each word based on its frequency
    sorted_dict = {key: rank for rank, key in enumerate(sorted(set(vocab.values()), reverse=True), 1)}
    vocab_rank = {k: sorted_dict[v] for k, v in vocab.items()}


    return {"vocab":vocab,
            "vocab_rank":vocab_rank,
            }


class CondGenExplainaboardBuilder:
    def __init__(
        self,
        info: SysOutputInfo,
        system_output_object: Iterable[dict],
        feature_table: Optional[Table] = {},
        gen_kwargs: dict = None,
    ):
        self._info = copy.deepcopy(info)
        self._system_output: Iterable[dict] = system_output_object
        self.gen_kwargs = gen_kwargs
        self._data: Table = feature_table
        # _samples_over_bucket_true: Dict(feature_name, bucket_name, sample_id_true_label):
        # samples in different buckets
        self._samples_over_bucket = {}
        # _performances_over_bucket: performance in different bucket: Dict(feature_name, bucket_name, performance)
        self._performances_over_bucket = {}
        self.score_dic = None

        # Calculate statistics of training set
        self.statistics = None
        if None != self._info.dataset_name:
            try:
                dataset = load_dataset(self._info.dataset_name, self._info.sub_dataset_name)
                if len(dataset['train']._stat) == 0 or self._info.reload_stat == False: # calculate the statistics (_stat) when _stat is {} or `reload_stat` is False
                    new_train = dataset['train'].apply(get_statistics, mode = "local")
                    self.statistics = new_train._stat
                else:
                    self.statistics = dataset["train"]._stat
            except FileNotFoundError as err:
                eprint(
                    "The dataset hasn't been supported by DataLab so no training set dependent features will be supported by ExplainaBoard."
                    "You can add the dataset by: https://github.com/ExpressAI/DataLab/blob/main/docs/SDK/add_new_datasets_into_sdk.md")




    @staticmethod
    def get_bucket_feature_value(feature_name: str):
        return "self._get_" + feature_name

    # training set dependent features
    def _get_num_oov(self, existing_features: dict):

        # exit()
        num_oov = 0

        for w in existing_features["source"].split(" "):  # should this be normalized for the consistency with DataLab?
            if w not in self.statistics['vocab'].keys():
                num_oov += 1
        # print(num_oov)
        return num_oov


    # training set dependent features (this could be merged into the above one for further optimization)
    def _get_fre_rank(self, existing_features: dict):
        fre_rank = 0

        for w in existing_features["source"].split(" "):
            if w not in self.statistics['vocab_rank'].keys():
                fre_rank += len(self.statistics['vocab_rank'])
            else:
                fre_rank += self.statistics['vocab_rank'][w]

        fre_rank = fre_rank * 1.0 / len(existing_features["source"].split(" "))
        return fre_rank


    def _complete_feature(self):
        """
        This function is used to calculate features used for bucekting, such as sentence_length
        :param feature_table_iterator:
        :return:
        """
        inputs = []
        for _id, feature_table in enumerate(self._system_output):
            inputs.append(
                {
                    "source": feature_table["source"],
                    "references": [feature_table["reference"]],
                    "hypothesis": feature_table["hypothesis"],
                }
            )
            self._data[_id] = feature_table

        self.score_dic = client.score(
            inputs, task="sum", metrics=self._info.metric_names.copy(), lang="en"
        )
        # print(self.score_dic["sample_level"][0].keys())

        # Get names of bucketing features
        # print(f"self._info.features.get_bucket_features()\n {self._info.features.get_bucket_features()}")
        bucket_features = self._info.features.get_bucket_features()

        for _id, dict_sysout in self._data.items():
            for bucket_feature in bucket_features:

                # this is need due to `del self._info.features[bucket_feature]`
                if bucket_feature not in self._info.features.keys():
                    continue
                # If there is a training set dependent feature while no pre-computed statistics for it,
                # then skip bucketing along this feature
                if self._info.features[bucket_feature].require_training_set and self.statistics == None:
                    del self._info.features[bucket_feature]
                    continue

                if bucket_feature in self.score_dic["sample_level"][_id].keys(): # features calculated from EaaS
                    feature_value = self.score_dic["sample_level"][_id][bucket_feature]
                    dict_sysout[
                        bucket_feature
                    ] = feature_value  # !!!!!!!!!!!!!!!!!!!! string to float !!!!!
                else:
                    feature_value = eval(
                        CondGenExplainaboardBuilder.get_bucket_feature_value(bucket_feature)
                    )(dict_sysout)

                    dict_sysout[bucket_feature] = feature_value

            self._data[_id] = dict_sysout
            yield _id, dict_sysout

    def get_overall_performance(self):

        inputs = []  # noqa
        metrics = self._info.metric_names  # noqa

        for metric_name in self._info.metric_names:

            overall_value = self.score_dic["corpus_level"]["corpus_" + metric_name]
            confidence_score_low = 0.0
            confidence_score_up = 0.0
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

        sample_address = ""
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

            # print(f"self._samples_over_bucket:\n{self._samples_over_bucket}")

            # evaluating bucket: get bucket performance
            self._performances_over_bucket[feature_name] = self.get_bucket_performance(
                feature_name
            )

    def get_bucket_performance(self, feature_name: str):
        """
        This function defines how to get bucket-level performance w.r.t a given feature (e.g., sentence length)
        :param feature_name: the name of a feature, e.g., sentence length
        :return: bucket_name_to_performance: a dictionary that maps bucket names to bucket performance
        """

        bucket_name_to_performance = {}
        for bucket_interval, sample_ids in self._samples_over_bucket[
            feature_name
        ].items():

            bucket_true_labels = []  # noqa
            bucket_predicted_labels = []  # noqa
            bucket_cases = []

            bucket_inputs = []
            dict_metric_to_values = {}

            for sample_id in sample_ids:

                source = self._data[int(sample_id)]["source"]
                reference = self._data[int(sample_id)]["reference"]
                hypothesis = self._data[int(sample_id)]["hypothesis"]

                bucket_inputs.append(
                    {
                        "source": source,
                        "references": [reference],
                        "hypothesis": hypothesis,
                    }
                )

                if self._info.results.is_print_case:
                    # #bucket_case =  reference + "|||" + hypothesis
                    # bucket_case = {"source": (sample_id, ["source"]),
                    #                "references": (sample_id, ["references"]),
                    #                "hypothesis": (sample_id, ["hypothesis"])}
                    bucket_case = str(sample_id)
                    bucket_cases.append(bucket_case)

                for metric_name in self._info.metric_names:
                    metric_value = self.score_dic["sample_level"][int(sample_id)][
                        metric_name
                    ]  # This would be modified later
                    if metric_name not in dict_metric_to_values.keys():
                        dict_metric_to_values[metric_name] = [metric_value]
                    else:
                        dict_metric_to_values[metric_name].append(metric_value)

            bucket_name_to_performance[bucket_interval] = []

            for metric_name in self._info.metric_names:

                bucket_value = numpy.average(dict_metric_to_values[metric_name])
                confidence_score_low = 0.0
                confidence_score_up = 0.0

                # print(
                #       f"value:\t {bucket_value}\n"
                #       f"confidence low\t {confidence_score_low}\n"
                #       f"confidence up \t {confidence_score_up}\n"
                #       f"---------------------------------")

                bucket_performance = BucketPerformance(
                    bucket_name=bucket_interval,
                    metric_name=metric_name,
                    value=format(bucket_value, '.4g'),
                    confidence_score_low=format(confidence_score_low, '.4g'),
                    confidence_score_up=format(confidence_score_up, '.4g'),
                    n_samples=len(dict_metric_to_values[metric_name]),
                    bucket_samples=bucket_cases,
                )

                bucket_name_to_performance[bucket_interval].append(bucket_performance)

        return sort_dict(bucket_name_to_performance)  # noqa

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

    def run(self):
        eb_generator = self._complete_feature()
        self._bucketing_samples(eb_generator)
        self.get_overall_performance()
        self._print_bucket_info()
        self._generate_report()
        return self._info
