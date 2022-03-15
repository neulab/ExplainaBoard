from explainaboard.builders import ExplainaboardBuilder
from explainaboard.utils.spacy_loader import spacy_loader


class ABSCExplainaboardBuilder(ExplainaboardBuilder):
    """
    Input: System Output file List[dict];  Metadata info
    Output: Analysis
    """

    def __init__(self):
        super().__init__()
        self._spacy_nlp = spacy_loader.get_model("en_core_web_sm")

    # --- Feature functions accessible by ExplainaboardBuilder._get_feature_func()
    def _get_sentence_length(self, existing_features: dict):
        return len(existing_features["text"].split(" "))

    def _get_token_number(self, existing_feature: dict):
        return len(existing_feature["text"])

    def _get_entity_number(self, existing_feature: dict):
        return len(self._spacy_nlp(existing_feature["text"]).ents)

    def _get_label(self, existing_feature: dict):
        return existing_feature["true_label"]

    def _get_aspect_length(self, existing_features: dict):
        return len(existing_features["aspect"].split(" "))

    def _get_aspect_index(self, existing_features: dict):
        return existing_features["text"].find(existing_features["aspect"])

    # --- End feature functions
