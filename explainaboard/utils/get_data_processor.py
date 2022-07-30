from datalabs.operations.featurize.nlp_featurize import NLPFeaturizing


def get_data_processor(feature_name, feature_func):
    def _processor(x):
        y = feature_func(x)
        return {feature_name: y}

    my_processor = NLPFeaturizing(func=_processor)
    return my_processor
