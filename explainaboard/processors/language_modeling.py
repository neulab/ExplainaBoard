from __future__ import annotations

from typing import Any

from datalabs import aggregating

from explainaboard import feature, TaskType
from explainaboard.info import BucketPerformance, Performance, SysOutputInfo
from explainaboard.metric import LogProbConfig, MetricConfig, MetricStats
from explainaboard.processors.processor import Processor
from explainaboard.processors.processor_registry import register_processor
from explainaboard.utils import bucketing
from explainaboard.utils.analysis import cap_feature
from explainaboard.utils.logging import progress
from explainaboard.utils.py_utils import sort_dict
from explainaboard.utils.tokenizer import Tokenizer
from explainaboard.utils.typing_utils import unwrap


@register_processor(TaskType.language_modeling)
class LanguageModelingProcessor(Processor):
    @classmethod
    def task_type(cls) -> TaskType:
        return TaskType.language_modeling

    @classmethod
    def default_features(cls) -> feature.Features:
        return feature.Features(
            {
                "text": feature.Value("string"),
                "log_probs": feature.Value("string"),
                "text_length": feature.Value(
                    dtype="float",
                    description="text length in tokens",
                    is_bucket=True,
                    bucket_info=feature.BucketInfo(
                        method="bucket_attribute_specified_bucket_value",
                        number=4,
                        setting=(),
                    ),
                ),
                "text_chars": feature.Value(
                    dtype="float",
                    description="text length in characters",
                    is_bucket=True,
                    bucket_info=feature.BucketInfo(
                        method="bucket_attribute_specified_bucket_value",
                        number=4,
                        setting=(),
                    ),
                ),
                "num_oov": feature.Value(
                    dtype="float",
                    description="the number of out-of-vocabulary words",
                    is_bucket=True,
                    bucket_info=feature.BucketInfo(
                        method="bucket_attribute_specified_bucket_value",
                        number=4,
                        setting=(),
                    ),
                    require_training_set=True,
                ),
                "fre_rank": feature.Value(
                    dtype="float",
                    description=(
                        "the average rank of each work based on its frequency in "
                        "training set"
                    ),
                    is_bucket=True,
                    bucket_info=feature.BucketInfo(
                        method="bucket_attribute_specified_bucket_value",
                        number=4,
                        setting=(),
                    ),
                    require_training_set=True,
                ),
                "length_fre": feature.Value(
                    dtype="float",
                    description="the frequency of text length in training set",
                    is_bucket=True,
                    bucket_info=feature.BucketInfo(
                        method="bucket_attribute_specified_bucket_value",
                        number=4,
                        setting=(),
                    ),
                    require_training_set=True,
                ),
                # --- the following are features of each token ---
                "tok_info": feature.Sequence(
                    feature=feature.Dict(
                        feature={
                            "tok_text": feature.Value("string"),
                            "tok_pos": feature.Position(positions=[0, 0]),
                            "tok_log_prob": feature.Value(
                                dtype="float",
                                description=(
                                    "log probability of the token according to the LM"
                                ),
                                is_bucket=False,
                            ),
                            "tok_capitalness": feature.Value(
                                dtype="string",
                                description=(
                                    "The capitalness of an token. For example, "
                                    "first_caps represents only the first character of "
                                    "the token is capital. full_caps denotes all "
                                    "characters of the token are capital"
                                ),
                                is_bucket=True,
                                bucket_info=feature.BucketInfo(
                                    method="bucket_attribute_discrete_value",
                                    number=4,
                                    setting=1,
                                ),
                            ),
                            "tok_position": feature.Value(
                                dtype="float",
                                description=(
                                    "The relative position of a token in a sentence"
                                ),
                                is_bucket=True,
                                bucket_info=feature.BucketInfo(
                                    method="bucket_attribute_specified_bucket_value",
                                    number=4,
                                    setting=(),
                                ),
                            ),
                            "tok_chars": feature.Value(
                                dtype="float",
                                description="The number of characters in a token",
                                is_bucket=True,
                                bucket_info=feature.BucketInfo(
                                    method="bucket_attribute_specified_bucket_value",
                                    number=4,
                                    setting=(),
                                ),
                            ),
                            "tok_test_freq": feature.Value(
                                dtype="float",
                                description="tok frequency in the test set",
                                is_bucket=True,
                                require_training_set=False,
                                bucket_info=feature.BucketInfo(
                                    method="bucket_attribute_specified_bucket_value",
                                    number=4,
                                    setting=(),
                                ),
                            ),
                            "tok_train_freq": feature.Value(
                                dtype="float",
                                description="tok frequency in the training set",
                                is_bucket=True,
                                require_training_set=True,
                                bucket_info=feature.BucketInfo(
                                    method="bucket_attribute_specified_bucket_value",
                                    number=4,
                                    setting=(),
                                ),
                            ),
                        }
                    )
                ),
            }
        )

    @classmethod
    def default_metrics(cls, language=None) -> list[MetricConfig]:
        return [
            LogProbConfig(name='Perplexity', language=language, ppl=True),
            LogProbConfig(name='LogProb', language=language, ppl=False),
        ]

    def _get_true_label(self, data_point: dict):
        return None

    def _get_predicted_label(self, data_point: dict):
        return [float(x) for x in data_point["log_probs"].split(' ')]

    # training set dependent features
    def _get_num_oov(self, tokens, statistics):
        num_oov = 0
        for w in tokens:
            if w not in statistics['vocab']:
                num_oov += 1
        return num_oov

    # training set dependent features
    # (this could be merged into the above one for further optimization)
    def _get_fre_rank(self, tokens, statistics):
        vocab_stats = statistics['vocab_rank']
        fre_rank = 0.0
        for w in tokens:
            fre_rank += vocab_stats.get(w, len(vocab_stats))
        fre_rank = 0 if len(tokens) == 0 else fre_rank / len(tokens)
        return fre_rank

    # training set dependent features
    def _get_length_fre(
        self, sys_info: SysOutputInfo, existing_features: dict, statistics: Any
    ):
        length_fre = 0
        length = len(sys_info.tokenize(existing_features["text"]))

        if length in statistics['length_fre'].keys():
            length_fre = statistics['length_fre'][length]

        return length_fre

    # --- End feature functions

    def _complete_features(
        self, sys_info: SysOutputInfo, sys_output: list[dict], external_stats=None
    ) -> list[str]:
        """
        This function takes in meta-data about system outputs, system outputs, and a few
        other optional pieces of information, then calculates feature functions and
        modifies `sys_output` to add these feature values

        :param sys_info: Information about the system output
        :param sys_output: The system output itself
        :param external_stats: Training set statistics that are used to calculate
            training set specific features
        :return: The features that are active (e.g. skipping training set features when
            no training set available)
        """
        sys_features = unwrap(sys_info.features)
        active_features = list(
            sys_features.get_bucket_features(
                include_training_dependent=external_stats is not None
            )
        )

        # One pass over the test set to find token test frequency
        all_tokens = [sys_info.tokenize(x['output']) for x in sys_output]
        all_log_probs = [self._get_predicted_label(x) for x in sys_output]
        test_freq: dict[str, int] = {}
        for tokens in all_tokens:
            for tok in tokens:
                test_freq[tok] = test_freq.get(tok, 0) + 1

        sent_feats: list[str] = []
        tok_feats: list[str] = []
        for x in active_features:
            (sent_feats if (x in sys_features) else tok_feats).append(x)

        for _id, (dict_sysout, tokens, log_probs) in progress(
            enumerate(zip(sys_output, all_tokens, all_log_probs)), desc="featurizing"
        ):
            # Get values of bucketing features
            text = dict_sysout["output"]

            # sentence_length
            dict_sysout["text_length"] = len(tokens)
            dict_sysout["text_chars"] = len(text)

            # sentence-level training set dependent features
            if external_stats is not None:
                dict_sysout["num_oov"] = self._get_num_oov(tokens, external_stats)
                dict_sysout["fre_rank"] = self._get_fre_rank(tokens, external_stats)

            # span features for true and predicted spans
            dict_sysout["tok_info"] = self._complete_tok_features(
                tokens, log_probs, test_freq, statistics=external_stats
            )

        return active_features

    def _complete_tok_features(self, toks, log_probs, test_freq, statistics=None):

        # Get training set stats if they exist
        has_stats = statistics is not None and len(statistics) > 0
        fre_dic = statistics["vocab"] if has_stats else None

        tok_dics = []
        for i, (tok, log_prob) in enumerate(zip(toks, log_probs)):
            # Basic features
            tok_dic = {
                'tok_text': tok,
                'tok_pos': (i, i + 1),
                'tok_log_prob': log_prob,
                'tok_capitalness': cap_feature(tok),
                'tok_position': i * 1.0 / len(toks),
                'tok_chars': len(tok),
                'tok_test_freq': test_freq.get(tok, 0),
            }
            # Training set dependent features
            if has_stats:
                tok_dic['tok_train_freq'] = fre_dic.get(tok, 0)
            # Save the features
            tok_dics.append(tok_dic)

        return tok_dics

    def _get_feature_dict(
        self, sys_output: list[dict], feature_name: str
    ) -> dict[tuple[int, int], Any]:
        feat_dict = {}
        for sample_id, my_output in enumerate(sys_output):
            for tok_id, tok_info in enumerate(my_output['tok_info']):
                feat_dict[(sample_id, tok_id)] = tok_info[feature_name]
        return feat_dict

    def _bucketing_samples(
        self,
        sys_info: SysOutputInfo,
        sys_output: list[dict],
        active_features: list[str],
        metric_stats: list[MetricStats],
    ) -> tuple[dict, dict]:

        features = unwrap(sys_info.features)

        sent_feats: list[str] = []
        tok_feats: list[str] = []
        for x in active_features:
            (sent_feats if (x in features) else tok_feats).append(x)

        # First, get the buckets for sentences using the standard protocol
        samples_over_bucket, performances_over_bucket = super()._bucketing_samples(
            sys_info, sys_output, sent_feats, metric_stats
        )

        # Bucketing
        for feature_name in progress(tok_feats, desc="token-level bucketing"):
            my_feature = features["tok_info"].feature.feature[feature_name]
            bucket_info = my_feature.bucket_info

            # Get buckets for true spans
            bucket_func = getattr(bucketing, bucket_info.method)

            feat_dict = self._get_feature_dict(sys_output, feature_name)

            samples_over_bucket[feature_name] = bucket_func(
                dict_obj=feat_dict,
                bucket_number=bucket_info.number,
                bucket_setting=bucket_info.setting,
            )

            # evaluating bucket: get bucket performance
            performances_over_bucket[feature_name] = self.get_bucket_performance_lm(
                sys_info,
                sys_output,
                samples_over_bucket[feature_name],
            )
        return samples_over_bucket, performances_over_bucket

    def get_bucket_performance_lm(
        self,
        sys_info: SysOutputInfo,
        sys_output: list[dict],
        samples_over_bucket: dict[str, list[tuple[int, int]]],
    ) -> dict[str, list[BucketPerformance]]:
        """
        This function defines how to get bucket-level performance w.r.t a given feature
        (e.g., sentence length)
        :param sys_info: Information about the system output
        :param sys_output: The system output itself
        :param samples_over_bucket_true: a dictionary mapping bucket interval names to
            true sample IDs
        :param samples_over_bucket_pred: a dictionary mapping bucket interval names to
            predicted sample IDs
        :return: bucket_name_to_performance: a dictionary that maps bucket names to
            bucket performance
        """

        bucket_name_to_performance = {}
        for bucket_interval, tok_list in samples_over_bucket.items():

            bucket_metrics = [x.to_metric() for x in unwrap(sys_info.metric_configs)]
            log_probs = [
                sys_output[samp_id]['tok_info'][tok_id]['tok_log_prob']
                for (samp_id, tok_id) in tok_list
            ]

            bucket_performance = BucketPerformance(
                bucket_name=bucket_interval,
                n_samples=len(tok_list),
                bucket_samples=tok_list,
            )
            for metric in bucket_metrics:
                metric_val = metric.evaluate(
                    None, log_probs, conf_value=sys_info.conf_value
                )
                conf_low, conf_high = (
                    metric_val.conf_interval if metric_val.conf_interval else None
                )
                performance = Performance(
                    metric_name=metric.config.name,
                    value=metric_val.value,
                    confidence_score_low=conf_low,
                    confidence_score_high=conf_high,
                )
                bucket_performance.performances.append(performance)

            bucket_name_to_performance[bucket_interval] = bucket_performance

        return sort_dict(bucket_name_to_performance)

    @aggregating()
    def _statistics_func(self, samples, tokenizer: Tokenizer):
        """
        Input:
        samples: [{
         "text":
         "label":
        }]
        """

        vocab: dict[str, float] = {}
        length_fre: dict[int, float] = {}
        total_samps = 0
        for sample in progress(samples):
            text = sample["text"]
            tokens = tokenizer(text)
            length = len(tokens)

            length_fre[length] = length_fre.get(length, 0.0) + 1.0

            # update vocabulary
            for w in tokens:
                vocab[w] = vocab.get(w, 0.0) + 1.0

            total_samps += 1

        # the rank of each word based on its frequency
        sorted_dict = {
            key: rank
            for rank, key in enumerate(sorted(set(vocab.values()), reverse=True), 1)
        }
        vocab_rank = {k: sorted_dict[v] for k, v in vocab.items()}

        for k, v in length_fre.items():
            length_fre[k] = v * 1.0 / total_samps

        return {"vocab": vocab, "vocab_rank": vocab_rank, "length_fre": length_fre}
