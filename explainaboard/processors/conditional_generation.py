from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterator
from typing import Any

from datalabs import aggregating
from eaas.async_client import AsyncClient
import numpy as np

from explainaboard import feature, TaskType
from explainaboard.info import BucketPerformance, Performance, SysOutputInfo
from explainaboard.metric import (
    EaaSMetricConfig,
    F1ScoreConfig,
    MetricConfig,
    MetricStats,
)
from explainaboard.processors.processor import Processor
from explainaboard.processors.processor_registry import register_processor
from explainaboard.utils import bucketing
from explainaboard.utils.analysis import cap_feature
import explainaboard.utils.feature_funcs
from explainaboard.utils.logging import progress
from explainaboard.utils.py_utils import sort_dict
from explainaboard.utils.tokenizer import Tokenizer
from explainaboard.utils.typing_utils import unwrap, unwrap_generator


@register_processor(TaskType.conditional_generation)
class ConditionalGenerationProcessor(Processor):
    @classmethod
    def task_type(cls) -> TaskType:
        return TaskType.conditional_generation

    @classmethod
    def default_features(cls) -> feature.Features:
        return feature.Features(
            {
                "source": feature.Value("string"),
                "reference": feature.Value("string"),
                "hypothesis": feature.Value("string"),
                "source_length": feature.Value(
                    dtype="float",
                    description="the length of source document",
                    is_bucket=True,
                    bucket_info=feature.BucketInfo(
                        method="bucket_attribute_specified_bucket_value",
                        number=4,
                        setting=(),
                    ),
                ),
                "reference_length": feature.Value(
                    dtype="float",
                    description="the length of gold summary",
                    is_bucket=True,
                    bucket_info=feature.BucketInfo(
                        method="bucket_attribute_specified_bucket_value",
                        number=4,
                        setting=(),
                    ),
                ),
                "hypothesis_length": feature.Value(
                    dtype="float",
                    description="the length of gold summary",
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
                "src_fre_rank": feature.Value(
                    dtype="float",
                    description=(
                        "the average rank of each word in the source sentence based on "
                        "its frequency in training set"
                    ),
                    is_bucket=True,
                    bucket_info=feature.BucketInfo(
                        method="bucket_attribute_specified_bucket_value",
                        number=4,
                        setting=(),
                    ),
                    require_training_set=True,
                ),
                # --- the following are features of each token ---
                "ref_tok_info": feature.Sequence(
                    feature=feature.Dict(
                        feature={
                            "tok_text": feature.Value("string"),
                            "tok_pos": feature.Position(positions=[0, 0]),
                            "tok_matched": feature.Value(
                                # this is actually "int" but int is not supported
                                dtype="float",
                                description=(
                                    "which token the ref/hyp token matches in the "
                                    "hyp/ref sentence, or -1 if none"
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
        defaults = ['rouge1', 'rouge2', 'rougeL', 'bleu']
        return [EaaSMetricConfig(name=x, language=language) for x in defaults]

    @classmethod
    def full_metric_list(cls, language=None) -> list[MetricConfig]:
        full_metrics = [
            "bleu",
            "bart_score_summ",
            "bart_score_mt",
            "bart_score_cnn_hypo_ref",
            "rouge1",
            "rouge2",
            "rougeL",
            "bert_score_f",
            "bert_score_p",
            "bert_score_r",
            "chrf",
            "comet",
            "mover_score",
            "prism",
        ]
        return [EaaSMetricConfig(name=x, language=language) for x in full_metrics]

    # --- Feature functions accessible by ExplainaboardBuilder._get_feature_func()
    def _get_source_length(self, sys_info: SysOutputInfo, existing_features: dict):
        return len(sys_info.tokenize(existing_features["source"]))

    def _get_reference_length(self, sys_info: SysOutputInfo, existing_features: dict):
        return len(sys_info.tokenize(existing_features["reference"]))

    def _get_hypothesis_length(self, sys_info: SysOutputInfo, existing_features: dict):
        return len(sys_info.tokenize(existing_features["hypothesis"]))

    # training set dependent features (could be merged for optimization?)
    def _get_num_oov(
        self, sys_info: SysOutputInfo, existing_features: dict, statistics: Any
    ):
        return explainaboard.utils.feature_funcs.feat_num_oov(
            existing_features,
            statistics,
            lambda x: x['source'],
            unwrap(sys_info.tokenizer),
        )

    def _get_src_fre_rank(
        self, sys_info: SysOutputInfo, existing_features: dict, statistics: Any
    ):
        return explainaboard.utils.feature_funcs.feat_freq_rank(
            existing_features,
            statistics,
            lambda x: x['source'],
            unwrap(sys_info.tokenizer),
        )

    def _get_true_label(self, data_point: dict):
        return data_point["reference"]

    def _get_predicted_label(self, data_point: dict):
        return data_point["hypothesis"]

    def _gen_metric_stats(
        self, sys_info: SysOutputInfo, sys_output: list[dict]
    ) -> list[MetricStats]:
        """Generate sufficient statistics for scoring.
        :param sys_info: Information about the system outputs
        :param sys_output: The system output itself
        :return: Statistics sufficient for scoring
        """

        # Queue up EaaS client request for all metrics
        inputs = []
        for _id, feature_table in enumerate(sys_output):
            inputs.append(
                {
                    "source": feature_table["source"],
                    "references": [feature_table["reference"]],
                    "hypothesis": feature_table["hypothesis"],
                }
            )
        metric_names = [x.name for x in unwrap_generator(sys_info.metric_configs)]
        async_request = self._get_eaas_client().async_score(
            inputs,
            metrics=metric_names,
            calculate=['corpus', 'stats'],
        )

        # Share the request result with all stats functions
        return [
            explainaboard.metric.EaaSMetricStats(
                name=name, pos=i, eaas_request=async_request
            )
            for i, name in enumerate(metric_names)
        ]

    # TODO(odashi): Restructure this function (and EaaS client) to be type-safe.
    def _fetch_metric_stats(self, metric_stats: dict[str, Any]):
        """
        A utility function used to lazily fetch the actual scoring dict when it's
        necessary.
        """
        if 'request_id' in metric_stats:
            client: AsyncClient = unwrap(self._eaas_client)
            eaas_stats: dict[str, Any] = client.wait_and_get_result(
                metric_stats['request_id']
            )
            metric_stats.clear()
            for k, v in eaas_stats.items():
                metric_stats[k] = v

    def _get_feature_info(self, name: str):
        if name in self._features:
            return self._features[name]
        else:
            return self._features['ref_tok_info'][name]

    def _complete_features(
        self, sys_info: SysOutputInfo, sys_output: list[dict], external_stats=None
    ) -> list[str]:
        """
        This function is used to calculate features used for bucketing, such as
        sentence_length
        :return:
        """

        # One pass over the test set to find token test frequency
        ref_test_freq: dict[str, int] = {}
        src_test_freq: dict[str, int] = {}
        for dict_sysout in sys_output:
            for ref_tok in sys_info.tokenize(dict_sysout['reference']):
                ref_test_freq[ref_tok] = ref_test_freq.get(ref_tok, 0) + 1
            for src_tok in sys_info.tokenize(dict_sysout['source']):
                src_test_freq[src_tok] = src_test_freq.get(src_tok, 0) + 1

        sys_features = unwrap(sys_info.features)

        # Get names of bucketing features
        bucket_feature_funcs = {}
        active_features = list(
            sys_features.get_bucket_features(
                include_training_dependent=external_stats is not None
            )
        )
        for bucket_feature in active_features:
            if bucket_feature in sys_features:
                bucket_feature_funcs[bucket_feature] = (
                    self._get_feature_func(bucket_feature),
                    sys_features[bucket_feature].require_training_set,
                )

        for _id, dict_sysout in enumerate(sys_output):

            # oracle_feats = self.get_oracle(dict_sysout, statistics)
            # Get values of bucketing features
            for (
                bucket_key,
                (
                    bucket_func,
                    training_dependent,
                ),
            ) in bucket_feature_funcs.items():
                # TODO(pengfei): should check the feature value type
                if training_dependent:
                    dict_sysout[bucket_key] = bucket_func(
                        sys_info, dict_sysout, external_stats
                    )
                else:
                    dict_sysout[bucket_key] = bucket_func(sys_info, dict_sysout)

            # span features for true and predicted spans
            ref_toks = sys_info.tokenize(dict_sysout['reference'])
            hyp_toks = sys_info.tokenize(dict_sysout['hypothesis'])
            dict_sysout["ref_tok_info"] = self._complete_tok_features(
                ref_toks, hyp_toks, ref_test_freq, statistics=external_stats
            )
            dict_sysout["hyp_tok_info"] = self._complete_tok_features(
                hyp_toks, ref_toks, ref_test_freq, statistics=external_stats
            )

        return active_features

    def _complete_tok_features(self, toks, other_toks, ref_test_freq, statistics=None):

        # Get training set stats if they exist
        has_stats = statistics is not None and len(statistics) > 0
        fre_dic = statistics["vocab"] if has_stats else None

        # Find tokens in other set
        other_tok_list = defaultdict(list)
        for i, tok in enumerate(other_toks):
            other_tok_list[tok].append(i)

        tok_dics = []
        for i, tok in enumerate(toks):
            # Basic features
            my_other = other_tok_list.get(tok, list())
            matched = my_other.pop(0) if len(my_other) > 0 else -1
            tok_dic = {
                'tok_text': tok,
                'tok_pos': (i, i + 1),
                'tok_matched': matched,
                'tok_capitalness': cap_feature(tok),
                'tok_position': i * 1.0 / len(toks),
                'tok_chars': len(tok),
                'tok_test_freq': ref_test_freq.get(tok, 0),
            }
            # Training set dependent features
            if has_stats:
                tok_dic['tok_train_freq'] = fre_dic.get(tok, 0)
            # Save the features
            tok_dics.append(tok_dic)

        return tok_dics

    def _get_feature_dict(
        self, sys_output: list[dict], feature_name: str, output_to_toks: Callable
    ):
        feat_dict = {}
        for samp_id, my_output in enumerate(sys_output):
            for tok_id, tok_info in enumerate(output_to_toks(my_output)):
                feat_dict[(samp_id, tok_id)] = tok_info[feature_name]
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
        samples_over_bucket_pred = {}

        # Second, get the buckets for tokens
        for feature_name in progress(tok_feats, desc="bucketing token features"):

            # Choose behavior based on whether this is a feature of samples or spans
            my_feature = features["ref_tok_info"].feature.feature[feature_name]
            bucket_info = my_feature.bucket_info

            # Get buckets for true spans
            bucket_func = getattr(bucketing, bucket_info.method)

            feat_dict = self._get_feature_dict(
                sys_output, feature_name, lambda x: x['ref_tok_info']
            )
            samples_over_bucket[feature_name] = bucket_func(
                dict_obj=feat_dict,
                bucket_number=bucket_info.number,
                bucket_setting=bucket_info.setting,
            )

            # Get buckets for predicted spans
            feat_dict = self._get_feature_dict(
                sys_output, feature_name, lambda x: x['hyp_tok_info']
            )
            samples_over_bucket_pred[
                feature_name
            ] = bucketing.bucket_attribute_specified_bucket_interval(
                dict_obj=feat_dict,
                bucket_number=bucket_info.number,
                bucket_setting=samples_over_bucket[feature_name].keys(),
            )

            # evaluating bucket: get bucket performance
            performances_over_bucket[feature_name] = self.get_bucket_performance_tok(
                sys_info,
                sys_output,
                samples_over_bucket[feature_name],
                samples_over_bucket_pred[feature_name],
            )
        return samples_over_bucket, performances_over_bucket

    def get_bucket_performance_tok(
        self,
        sys_info: SysOutputInfo,
        sys_output: list[dict],
        samples_over_bucket_true: dict[str, list[tuple[int, int]]],
        samples_over_bucket_pred: dict[str, list[tuple[int, int]]],
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

        bucket_name_to_performance: dict[str, BucketPerformance] = {}
        f1_score = F1ScoreConfig(name='F1', separate_match=True).to_metric()
        for bucket_interval, toks_true in samples_over_bucket_true.items():

            if bucket_interval not in samples_over_bucket_pred.keys():
                raise ValueError("Predict Label Bucketing Errors")
            else:
                toks_pred = samples_over_bucket_pred[bucket_interval]

            stats_list = []
            for sid, tid in toks_true:
                matched = (
                    1.0
                    if sys_output[sid]['ref_tok_info'][tid]['tok_matched'] >= 0
                    else 0.0
                )
                stats_list.append([1.0, 0.0, matched, 0.0])
            for sid, tid in toks_pred:
                matched = (
                    1.0
                    if sys_output[sid]['hyp_tok_info'][tid]['tok_matched'] >= 0
                    else 0.0
                )
                stats_list.append([0.0, 1.0, 0.0, matched])

            stats = explainaboard.metric.MetricStats(np.array(stats_list))
            result = f1_score.evaluate_from_stats(stats, conf_value=0.05)
            conf_interval: tuple[float, float] = unwrap(result.conf_interval)
            conf_low, conf_high = conf_interval
            performance = Performance(
                metric_name='F1',
                value=result.value,
                confidence_score_low=conf_low,
                confidence_score_high=conf_high,
            )
            bucket_performance = BucketPerformance(
                bucket_name=bucket_interval,
                n_samples=len(toks_true),
                bucket_samples=toks_true,
                performances=[performance],
            )
            bucket_name_to_performance[bucket_interval] = bucket_performance

        return sort_dict(bucket_name_to_performance)

    @aggregating()
    def _statistics_func(self, samples: Iterator, tokenizer: Tokenizer):
        return explainaboard.utils.feature_funcs.accumulate_vocab_from_samples(
            samples, lambda x: x['reference'], tokenizer
        )
