from typing import Callable, Iterator, Any

from datalabs import load_dataset
from datalabs.operations.aggregate.text_matching import text_matching_aggregating

from explainaboard.builders import ExplainaboardBuilder
from explainaboard.info import SysOutputInfo
from explainaboard.utils.analysis import *
from explainaboard.utils.feature_funcs import get_similarity_by_sacrebleu


@text_matching_aggregating(
    name="get_statistics",
    contributor="datalab",
    task="text-matching, natural-language-inference",
    description="Calculate the overall statistics (e.g., average length) of a given "
    "text pair classification datasets. e,g. natural language inference",
)
def get_statistics(samples: Iterator):
    """
    Input:
    samples: [{
     "text1":
     "text2":
     "label":
    }]
    """
    return ExplainaboardBuilder.accumulate_vocab_from_samples(samples, lambda x: x['text1']+x['text2'])


class TextPairClassificationExplainaboardBuilder(ExplainaboardBuilder):
    """
    Input: System Output file List[dict];  Metadata info
    Output: Analysis
    """

    def __init__(self):
        super().__init__()
        self._statistics_func = get_statistics

    def _init_statistics(self, sys_info: SysOutputInfo, statistics_func: Callable):
        """Take in information about the system outputs and a statistic calculating function and return a dictionary
        of statistics.

        :param sys_info: Information about the system outputs
        :param statistics_func: The function used to get the statistics
        :return: Statistics from, usually, the training set that are used to calculate other features
        """
        # TODO(gneubig): this should be deduplicated
        # Calculate statistics of training set
        self.statistics = None
        if sys_info.dataset_name is not None:
            try:
                dataset = load_dataset(sys_info.dataset_name, sys_info.sub_dataset_name)
                if (
                    len(dataset['train']._stat) == 0
                    or not sys_info.reload_stat
                ):  # calculate the statistics (_stat) when _stat is {} or `reload_stat` is False
                    new_train = dataset['train'].apply(statistics_func, mode="local")
                    self.statistics = new_train._stat
                else:
                    self.statistics = dataset["train"]._stat
            except FileNotFoundError:
                eprint(
                    "The dataset hasn't been supported by DataLab so no training set dependent features will be supported by ExplainaBoard." # noqa
                    "You can add the dataset by: https://github.com/ExpressAI/DataLab/blob/main/docs/SDK/add_new_datasets_into_sdk.md" # noqa
                )

    # --- Feature functions accessible by ExplainaboardBuilder._get_feature_func()
    def _get_similarity(self, existing_features: dict):
        return get_similarity_by_sacrebleu(
            existing_features["text1"], existing_features["text2"]
        )

    def _get_text1_length(self, existing_features: dict):
        return len(existing_features["text1"].split(" "))

    def _get_text2_length(self, existing_feature: dict):
        return len(existing_feature["text2"])

    def _get_text1_divided_text2(self, existing_feature: dict):
        return len(existing_feature["text1"]) * 1.0 / len(existing_feature["text2"])

    def _get_label(self, existing_feature: dict):
        # print(f"print_existing_feature: \t {existing_feature}")
        return existing_feature["true_label"]

    # training set dependent features
    def _get_num_oov(self, existing_features: dict, statistics: Any):
        return ExplainaboardBuilder.feat_num_oov(existing_features, statistics, lambda x: x['text1'] + x['text2'])

    # training set dependent features (this could be merged into the above one for further optimization)
    def _get_fre_rank(self, existing_features: dict, statistics: Any):
        return ExplainaboardBuilder.feat_freq_rank(existing_features, statistics, lambda x: x['text1'] + x['text2'])

    # --- End feature functions
