from typing import Callable
from explainaboard.info import SysOutputInfo, BucketPerformance, Performance, Table
from explainaboard.builders import ExplainaboardBuilder
from explainaboard.utils.eval_bucket import *
from explainaboard.utils.feature_funcs import get_similarity_by_sacrebleu
from explainaboard.utils.analysis import *
from explainaboard.metric import *
from tqdm import tqdm

from typing import Iterator
from datalabs import load_dataset
from datalabs.operations.aggregate.text_matching import text_matching_aggregating


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

    vocab = {}
    length_fre = {}
    for sample in tqdm(samples):

        text1, text2, label = sample["text1"], sample["text2"], sample["label"]
        # length = len(text.split(" "))

        # if length in length_fre.keys():
        #     length_fre[length] += 1
        # else:
        #     length_fre[length] = 1

        # update vocabulary
        for w in (text1 + text2).split(" "):
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

    # for k, v in length_fre.items():
    #     length_fre[k] /= len(samples)

    return {
        "vocab": vocab,
        "vocab_rank": vocab_rank,
        # "length_fre":length_fre,
    }


class TextPairClassificationExplainaboardBuilder(ExplainaboardBuilder):
    """
    Input: System Output file List[dict];  Metadata info
    Output: Analysis
    """

    def __init__(self):
        super().__init__()

    def _init_statistics(self,
                         sys_info: SysOutputInfo,
                         get_statistics: Callable):
        """Take in information about the system outputs and a statistic calculating function and return a dictionary
        of statistics.

        :param sys_info: Information about the system outputs
        :param get_statistics: The function used to get the statistics
        :return: Statistics from, usually, the training set that are used to calculate other features
        """
        # TODO(gneubig): this should be deduplicated
        # Calculate statistics of training set
        self.statistics = None
        if None != sys_info.dataset_name:
            try:
                dataset = load_dataset(
                    sys_info.dataset_name, sys_info.sub_dataset_name
                )
                if (
                    len(dataset['train'], BucketPerformance, Performance, Table._stat) == 0 or sys_info.reload_stat == False
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
    def _get_num_oov(self, existing_features: dict):
        num_oov = 0

        for w in (existing_features["text1"] + existing_features["text2"]).split(" "):
            if w not in self.statistics['vocab'].keys():
                num_oov += 1
        # print(num_oov)
        return num_oov

    # training set dependent features (this could be merged into the above one for further optimization)
    def _get_fre_rank(self, existing_features: dict):
        fre_rank = 0

        for w in (existing_features["text1"] + existing_features["text2"]).split(" "):
            if w not in self.statistics['vocab_rank'].keys():
                fre_rank += len(self.statistics['vocab_rank'])
            else:
                fre_rank += self.statistics['vocab_rank'][w]

        fre_rank = (
            fre_rank
            * 1.0
            / len((existing_features["text1"] + existing_features["text2"]).split(" "))
        )
        return fre_rank

    # --- End feature functions
