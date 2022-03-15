from typing import Callable, Any
from typing import Iterator

from datalabs import load_dataset
from datalabs.operations.aggregate.qa_multiple_choice import (
    qa_multiple_choice_aggregating,
)

from explainaboard.builders import ExplainaboardBuilder
from explainaboard.info import SysOutputInfo
from explainaboard.utils.analysis import eprint


@qa_multiple_choice_aggregating(
    name="get_statistics",
    contributor="datalab",
    task="qa-multiple-choice",
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
    return ExplainaboardBuilder.accumulate_vocab_from_samples(samples, lambda x: x['context'])


class QAMultipleChoiceExplainaboardBuilder(ExplainaboardBuilder):
    """
    Input: System Output file List[dict];  Metadata info
    Output: Analysis
    """

    def __init__(self):
        super().__init__()
        self._statistics_func = get_statistics

    # TODO(gneubig): this should be deduplicated
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
                if (
                    len(dataset['train']._stat) == 0 or not sys_info.reload_stat
                ):  # calculate the statistics (_stat) when _stat is {} or `reload_stat` is False
                    new_train = dataset['train'].apply(statistics_func, mode="local")
                    statistics = new_train._stat
                else:
                    statistics = dataset["train"]._stat
            except FileNotFoundError:
                eprint(
                    "The dataset hasn't been supported by DataLab so no training set dependent features will be supported by ExplainaBoard." # noqa
                    "You can add the dataset by: https://github.com/ExpressAI/DataLab/blob/main/docs/SDK/add_new_datasets_into_sdk.md" # noqa
                )
        return statistics

    # --- Feature functions accessible by ExplainaboardBuilder._get_feature_func()
    def _get_context_length(self, existing_features: dict):
        return len(existing_features["context"].split(" "))

    def _get_question_length(self, existing_features: dict):
        return len(existing_features["question"].split(" "))

    def _get_answer_length(self, existing_features: dict):
        return len(existing_features["answers"]["text"].split(" "))

    # training set dependent features
    def _get_num_oov(self, existing_features: dict, statistics: Any):
        return ExplainaboardBuilder.feat_num_oov(existing_features, statistics, lambda x: x['context'])

    # training set dependent features (this could be merged into the above one for further optimization)
    def _get_fre_rank(self, existing_features: dict, statistics: Any):
        return ExplainaboardBuilder.feat_freq_rank(existing_features, statistics, lambda x: x['context'])

    # --- End feature functions

    def _get_true_label(self, data_point):
        """
        Get the true label from a data point. Overloaded from parent class.
        :param data_point: the data point under consideration
        :return: the true label for the output
        """
        return data_point["answers"]["option_index"]

    def _get_predicted_label(self, data_point):
        """
        Get the predicted label from a data point. Overloaded from parent class.
        :param data_point: the data point under consideration
        :return: the predicted label for the output
        """
        return data_point["predicted_answers"]["option_index"]
