from typing import Iterable, Optional
from explainaboard.info import SysOutputInfo, BucketPerformance, Performance, Table
from explainaboard.utils import analysis
from explainaboard.builders import ExplainaboardBuilder
from explainaboard.utils.eval_bucket import *  # noqa
from explainaboard.utils.analysis import *  # noqa
from explainaboard.metric import *  # noqa
from tqdm import tqdm
from typing import Iterator, Dict, List
from datalabs import load_dataset
from datalabs.operations.aggregate.qa_multiple_choice import (
    qa_multiple_choice_aggregating,
)


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

    vocab = {}
    length_fre = {}
    for sample in tqdm(samples):
        context, answers, options = (
            sample["context"],
            sample["answers"],
            sample["options"],
        )

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


class QAMultipleChoiceExplainaboardBuilder(ExplainaboardBuilder):
    """
    Input: System Output file List[dict];  Metadata info
    Output: Analysis
    """

    def __init__(
        self,
        info: SysOutputInfo,
        system_output_object: Iterable[dict],
        feature_table: Optional[Table] = None,
        user_defined_feature_config=None,
    ):
        super().__init__(
            info, system_output_object, feature_table, user_defined_feature_config
        )

        # TODO(gneubig): this should be deduplicated
        # Calculate statistics of training set
        self.statistics = None
        if None != self._info.dataset_name:
            try:
                dataset = load_dataset(
                    self._info.dataset_name, self._info.sub_dataset_name
                )
                if (
                    len(dataset['train']._stat) == 0 or self._info.reload_stat == False
                ):  # calculate the statistics (_stat) when _stat is {} or `reload_stat` is False
                    new_train = dataset['train'].apply(get_statistics, mode="local")
                    self.statistics = new_train._stat
                else:
                    self.statistics = dataset["train"]._stat
            except FileNotFoundError as err:
                eprint(
                    "The dataset hasn't been supported by DataLab so no training set dependent features will be supported by ExplainaBoard."
                    "You can add the dataset by: https://github.com/ExpressAI/DataLab/blob/main/docs/SDK/add_new_datasets_into_sdk.md"
                )

    # --- Feature functions accessible by ExplainaboardBuilder._get_feature_func()
    def _get_context_length(self, existing_features: dict):
        return len(existing_features["context"].split(" "))

    def _get_question_length(self, existing_features: dict):
        return len(existing_features["question"].split(" "))

    def _get_answer_length(self, existing_features: dict):
        return len(existing_features["answers"]["text"].split(" "))

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