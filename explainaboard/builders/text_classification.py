from typing import Iterator

from tqdm import tqdm

from explainaboard.builders import ExplainaboardBuilder
from explainaboard.utils.feature_funcs import *
from explainaboard.utils.spacy_loader import spacy_loader
from datalabs.operations.aggregate.text_classification import text_classification_aggregating

@text_classification_aggregating(
    name="get_statistics",
    contributor="datalab",
    task="text-classification",
    description="Calculate the overall statistics (e.g., density) of a given text classification dataset",
)
def get_statistics(samples: Iterator):
    """
    Input:
    samples: [{
     "text":
     "label":
    }]
    """

    vocab = {}
    length_fre = {}
    for sample in tqdm(samples):
        text, label = sample["text"], sample["label"]
        length = len(text.split(" "))

        if length in length_fre.keys():
            length_fre[length] += 1
        else:
            length_fre[length] = 1

        # update vocabulary
        for w in text.split(" "):
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

    for k, v in length_fre.items():
        length_fre[k] = length_fre[k] * 1.0 / len(samples)

    return {"vocab": vocab, "vocab_rank": vocab_rank, "length_fre": length_fre}


class TCExplainaboardBuilder(ExplainaboardBuilder):
    """
    Input: System Output file List[dict];  Metadata info
    Output: Analysis
    """

    def __init__(self):
        super().__init__()
        self._statistics_func = get_statistics

    # --- Feature functions accessible by ExplainaboardBuilder._get_feature_func()
    def _get_sentence_length(self, existing_features: dict):
        return len(existing_features["text"].split(" "))

    def _get_token_number(self, existing_feature: dict):
        return len(existing_feature["text"])

    def _get_entity_number(self, existing_feature: dict):
        return len(
            spacy_loader.get_model("en_core_web_sm")(existing_feature["text"]).ents
        )

    def _get_label(self, existing_feature: dict):
        return existing_feature["true_label"]

    def _get_basic_words(self, existing_feature: dict):
        return get_basic_words(existing_feature["text"])  # noqa

    def _get_lexical_richness(self, existing_feature: dict):
        return get_lexical_richness(existing_feature["text"])  # noqa

    # training set dependent features
    def _get_num_oov(self, existing_features: dict):
        num_oov = 0

        for w in existing_features["text"].split(" "):
            if w not in self.statistics['vocab'].keys():
                num_oov += 1
        # print(num_oov)
        return num_oov

    # training set dependent features (this could be merged into the above one for further optimization)
    def _get_fre_rank(self, existing_features: dict):
        fre_rank = 0

        for w in existing_features["text"].split(" "):
            if w not in self.statistics['vocab_rank'].keys():
                fre_rank += len(self.statistics['vocab_rank'])
            else:
                fre_rank += self.statistics['vocab_rank'][w]

        fre_rank = fre_rank * 1.0 / len(existing_features["text"].split(" "))
        return fre_rank

    # training set dependent features
    def _get_length_fre(self, existing_features: dict):
        length_fre = 0
        length = len(existing_features["text"].split(" "))

        if length in self.statistics['length_fre'].keys():
            length_fre = self.statistics['length_fre'][length]

        return length_fre

    # --- End feature functions
