from typing import Callable
from explainaboard.info import SysOutputInfo, BucketPerformance, Performance, Table
from explainaboard.utils import analysis
from explainaboard.builders import ExplainaboardBuilder
from explainaboard.utils.eval_bucket import *  # noqa
from explainaboard.utils.analysis import *  # noqa
from explainaboard.utils.eval_basic_qa import *  # noqa
from explainaboard.metric import *  # noqa
from tqdm import tqdm


from typing import Iterator, Dict, List
from datalabs import load_dataset

from datalabs.operations.aggregate.qa_extractive import qa_extractive_aggregating


@qa_extractive_aggregating(
    name="get_statistics",
    contributor="datalab",
    task="qa-extractive",
    description="Calculate the overall statistics (e.g., average length) of "
    "a given text classification dataset",
)
def get_statistics(samples: Iterator):
    """
    Input:
    samples: [{
     "id":str
     "context":str
     "question":str
     "answers":Dict
     "options"
    }]
    """

    vocab = {}
    length_fre = {}
    for sample in tqdm(samples):
        context, answers = sample["context"], sample["answers"]

        # update vocabulary
        for w in context.split(" "):
            if w in vocab.keys():
                vocab[w] += 1
            else:
                vocab[w] = 1

    # the rank of each word based on its frequency
    sorted_dict = {
        key: rank
        for rank, key in enumerate(sorted(set(vocab.values()), reverse=True), 1)
    }
    vocab_rank = {k: sorted_dict[v] for k, v in vocab.items()}

    # print(vocab)
    # print(vocab_rank)
    # exit()

    return {
        "vocab": vocab,
        "vocab_rank": vocab_rank,
    }


class QAExtractiveExplainaboardBuilder(ExplainaboardBuilder):
    def __init__(self):
        super().__init__()

    # TODO(gneubig) to be deduplicated
    def _init_statistics(self,
                         sys_info: SysOutputInfo,
                         get_statistics: Callable):
        """Take in information about the system outputs and a statistic calculating function and return a dictionary
        of statistics.

        :param sys_info: Information about the system outputs
        :param get_statistics: The function used to get the statistics
        :return: Statistics from, usually, the training set that are used to calculate other features
        """

        # Calculate statistics of training set
        statistics = None
        if None != sys_info.dataset_name:
            try:
                dataset = load_dataset(
                    sys_info.dataset_name, sys_info.sub_dataset_name
                )
                if "train" not in dataset.keys():
                    statistics = None
                elif (
                    len(dataset['train']._stat) == 0 or sys_info.reload_stat == False
                ):  # calculate the statistics (_stat) when _stat is {} or `reload_stat` is False
                    new_train = dataset['train'].apply(get_statistics, mode="local")
                    statistics = new_train._stat
                else:
                    statistics = dataset["train"]._stat
            except FileNotFoundError as err:
                eprint(
                    "The dataset hasn't been supported by DataLab so no training set dependent features will be supported by ExplainaBoard."
                    "You can add the dataset by: https://github.com/ExpressAI/DataLab/blob/main/docs/SDK/add_new_datasets_into_sdk.md"
                )
        return statistics

    # --- Feature functions accessible by ExplainaboardBuilder._get_feature_func()
    def _get_context_length(self, existing_features: dict):
        return len(existing_features["context"].split(" "))

    def _get_question_length(self, existing_features: dict):
        return len(existing_features["question"].split(" "))

    def _get_answer_length(self, existing_features: dict):
        if isinstance(existing_features["answers"]["text"], list):
            return len(existing_features["answers"]["text"][0].split(" "))
        else:
            return len(existing_features["answers"]["text"].split(" "))

    def _get_sim_context_question(self, existing_features: dict):

        references = existing_features["context"]
        hypothesis = existing_features["question"]

        res_json = self._get_eaas_client().bleu([[references]], [hypothesis], lang="en")
        return res_json["corpus_bleu"]

    # training set dependent features
    def _get_num_oov(self, existing_features: dict):
        num_oov = 0

        for w in existing_features["context"].split(" "):
            if w not in self.statistics['vocab'].keys():
                num_oov += 1
        # print(num_oov)
        return num_oov

    # training set dependent features (this could be merged into the above one for further optimization)
    def _get_fre_rank(self, existing_features: dict):
        fre_rank = 0

        for w in existing_features["context"].split(" "):
            if w not in self.statistics['vocab_rank'].keys():
                fre_rank += len(self.statistics['vocab_rank'])
            else:
                fre_rank += self.statistics['vocab_rank'][w]

        fre_rank = fre_rank * 1.0 / len(existing_features["context"].split(" "))
        return fre_rank

    # --- End feature functions

    # TODO(gneubig): this can probably be generalized as well
    def get_overall_performance(self):
        predicted_answers, true_answers = [], []

        for _id, feature_table in self._data.items():
            predicted_answers.append(feature_table["predicted_answers"]["text"])
            true_answers.append(feature_table["answers"]["text"])

        for metric_name in self._info.metric_names:
            overall_value = eval(metric_name)(true_answers, predicted_answers)

            overall_value = overall_value
            confidence_score_low = 0
            confidence_score_up = 0
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

    # TODO(gneubig): this should be generalized
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

            bucket_true_labels = []
            bucket_predicted_labels = []
            bucket_cases = []

            for sample_id in sample_ids:

                true_label = self._data[int(sample_id)]["answers"]["text"]
                if isinstance(true_label, list):
                    true_label = true_label[0]

                predicted_label = self._data[int(sample_id)]["predicted_answers"][
                    "text"
                ]
                sent = self._data[int(sample_id)]["question"]  # noqa
                s_id = self._data[int(sample_id)]["id"]

                # get a bucket of true/predicted labels
                bucket_true_labels.append(true_label)
                bucket_predicted_labels.append(predicted_label)
                # get a bucket of cases (e.g., errors)
                if self._info.results.is_print_case:
                    if true_label != predicted_label:
                        # bucket_case = true_label[0] + "|||" + predicted_label + "|||" + sent
                        # bucket_case = {"true_answer": (sample_id, ["true_answers","text"]),
                        #                "predicted_answer": (sample_id, ["predicted_answer"]),
                        #                "question": (sample_id, ["question"])}
                        bucket_case = str(s_id)
                        bucket_cases.append(bucket_case)

            bucket_name_to_performance[bucket_interval] = []
            for metric_name in self._info.metric_names:

                bucket_value = eval(metric_name)(
                    bucket_true_labels, bucket_predicted_labels
                )

                bucket_value = bucket_value
                confidence_score_low = 0
                confidence_score_up = 0

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
                    n_samples=len(bucket_true_labels),
                    bucket_samples=bucket_cases,
                )

                bucket_name_to_performance[bucket_interval].append(bucket_performance)

        return sort_dict(bucket_name_to_performance)  # noqa
