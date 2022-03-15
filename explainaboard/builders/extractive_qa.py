from typing import Callable, Any
from typing import Iterator, Dict, List

from datalabs import load_dataset
from datalabs.operations.aggregate.qa_extractive import qa_extractive_aggregating

from explainaboard.builders import ExplainaboardBuilder
from explainaboard.info import SysOutputInfo, BucketPerformance, Performance
from explainaboard.utils.py_utils import eprint, sort_dict
import explainaboard.utils.eval_basic_qa


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
    return ExplainaboardBuilder.accumulate_vocab_from_samples(
        samples, lambda x: x['context']
    )


class QAExtractiveExplainaboardBuilder(ExplainaboardBuilder):
    def __init__(self):
        super().__init__()
        self._statistics_func = get_statistics

    # TODO(gneubig) to be deduplicated
    def _init_statistics(self, sys_info: SysOutputInfo, statistics_func: Callable):
        """Take in information about the system outputs and a statistic calculating function and return a dictionary
        of statistics.

        :param sys_info: Information about the system outputs
        :param statistics_func: The function used to get the statistics
        :return: Statistics from, usually, the training set that are used to calculate other features
        """

        # Calculate statistics of training set
        statistics = None
        if sys_info.dataset_name is not None:
            try:
                dataset = load_dataset(sys_info.dataset_name, sys_info.sub_dataset_name)
                if "train" not in dataset.keys():
                    statistics = None
                elif (
                    len(dataset['train']._stat) == 0 or not sys_info.reload_stat
                ):  # calculate the statistics (_stat) when _stat is {} or `reload_stat` is False
                    new_train = dataset['train'].apply(statistics_func, mode="local")
                    statistics = new_train._stat
                else:
                    statistics = dataset["train"]._stat
            except FileNotFoundError:
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

    # training set dependent features (could be merged for optimization?)
    def _get_num_oov(self, existing_features: dict, statistics: Any):
        return ExplainaboardBuilder.feat_num_oov(
            existing_features, statistics, lambda x: x['context']
        )

    def _get_fre_rank(self, existing_features: dict, statistics: Any):
        return ExplainaboardBuilder.feat_freq_rank(
            existing_features, statistics, lambda x: x['context']
        )

    # --- End feature functions

    # TODO(gneubig): this can probably be generalized as well
    def get_overall_performance(
        self,
        sys_info: SysOutputInfo,
        sys_output: List[dict],
    ) -> Dict[str, Performance]:
        predicted_answers, true_answers = [], []

        for _id, feature_table in enumerate(sys_output):
            predicted_answers.append(feature_table["predicted_answers"]["text"])
            true_answers.append(feature_table["answers"]["text"])

        overall = {}
        for metric_name in sys_info.metric_names:
            # TODO(gneubig): is it necessary to have this as a separate interface than the other metrics?
            #                probably not. it'd be good to unify these.
            metric_func = getattr(explainaboard.utils.eval_basic_qa, metric_name)
            overall_value = metric_func(true_answers, predicted_answers)

            overall_value = overall_value
            overall_performance = Performance(
                metric_name=metric_name,
                value=overall_value,
                confidence_score_low=None,
                confidence_score_high=None,
            )
            overall[metric_name] = overall_performance
        return overall

    # TODO(gneubig): this should be generalized
    def get_bucket_performance(
        self,
        sys_info: SysOutputInfo,
        sys_output: List[dict],
        samples_over_bucket: Dict[str, List[int]],
    ) -> Dict[str, List[BucketPerformance]]:
        """
        This function defines how to get bucket-level performance w.r.t a given feature (e.g., sentence length)
        :param sys_info: Information about the system output
        :param sys_output: The system output itself
        :param samples_over_bucket: a dictionary mapping bucket interval names to sample IDs for that bucket
        :return: bucket_name_to_performance: a dictionary that maps bucket names to bucket performance
        """

        bucket_name_to_performance = {}
        for bucket_interval, sample_ids in samples_over_bucket.items():

            bucket_true_labels = []
            bucket_predicted_labels = []
            bucket_cases = []

            for sample_id in sample_ids:

                true_label = sys_output[int(sample_id)]["answers"]["text"]
                if isinstance(true_label, list):
                    true_label = true_label[0]

                predicted_label = sys_output[int(sample_id)]["predicted_answers"][
                    "text"
                ]
                s_id = sys_output[int(sample_id)]["id"]

                # get a bucket of true/predicted labels
                bucket_true_labels.append(true_label)
                bucket_predicted_labels.append(predicted_label)
                # get a bucket of cases (e.g., errors)
                if sys_info.is_print_case:
                    if true_label != predicted_label:
                        # bucket_case = true_label[0] + "|||" + predicted_label + "|||" + sent
                        # bucket_case = {"true_answer": (sample_id, ["true_answers","text"]),
                        #                "predicted_answer": (sample_id, ["predicted_answer"]),
                        #                "question": (sample_id, ["question"])}
                        bucket_case = str(s_id)
                        bucket_cases.append(bucket_case)

            bucket_name_to_performance[bucket_interval] = []
            for metric_name in sys_info.metric_names:
                # TODO(gneubig): is it necessary to have this as a separate interface than the other metrics?
                #                probably not. it'd be good to unify these.
                metric_func = getattr(explainaboard.utils.eval_basic_qa, metric_name)
                bucket_value = metric_func(bucket_true_labels, bucket_predicted_labels)

                bucket_performance = BucketPerformance(
                    bucket_name=bucket_interval,
                    metric_name=metric_name,
                    value=bucket_value,
                    confidence_score_low=None,
                    confidence_score_high=None,
                    n_samples=len(bucket_true_labels),
                    bucket_samples=bucket_cases,
                )

                bucket_name_to_performance[bucket_interval].append(bucket_performance)

        return sort_dict(bucket_name_to_performance)
