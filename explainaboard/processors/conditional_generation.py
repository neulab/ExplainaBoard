from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterator
from typing import Any, cast

from datalabs import aggregating
from eaas.async_client import AsyncClient
import numpy as np

from explainaboard import feature, TaskType
from explainaboard.info import (
    BucketCase,
    BucketCaseCollection,
    BucketCaseMultiSpan,
    BucketCaseSpan,
    BucketPerformance,
    Performance,
    SysOutputInfo,
)
from explainaboard.metric import (
    EaaSMetricConfig,
    F1ScoreConfig,
    MetricConfig,
    MetricStats,
)
from explainaboard.processors.processor import Processor
from explainaboard.processors.processor_registry import register_processor
from explainaboard.utils import bucketing
import explainaboard.utils.feature_funcs
from explainaboard.utils.feature_funcs import accumulate_vocab_from_samples, cap_feature
from explainaboard.utils.logging import progress
from explainaboard.utils.tokenizer import TokenSeq
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
                    description="length of the source",
                    is_bucket=True,
                    bucket_info=feature.BucketInfo(
                        method="bucket_attribute_specified_bucket_value",
                        number=4,
                        setting=(),
                    ),
                ),
                "reference_length": feature.Value(
                    dtype="float",
                    description="length of the reference",
                    is_bucket=True,
                    bucket_info=feature.BucketInfo(
                        method="bucket_attribute_specified_bucket_value",
                        number=4,
                        setting=(),
                    ),
                ),
                "hypothesis_length": feature.Value(
                    dtype="float",
                    description="length of the hypothesis",
                    is_bucket=True,
                    bucket_info=feature.BucketInfo(
                        method="bucket_attribute_specified_bucket_value",
                        number=4,
                        setting=(),
                    ),
                ),
                "src_num_oov": feature.Value(
                    dtype="float",
                    description="OOV words in the source",
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
                        "average training-set frequency rank of words in sentence"
                    ),
                    is_bucket=True,
                    bucket_info=feature.BucketInfo(
                        method="bucket_attribute_specified_bucket_value",
                        number=4,
                        setting=(),
                    ),
                    require_training_set=True,
                ),
                "ref_num_oov": feature.Value(
                    dtype="float",
                    description="number of OOV words in reference",
                    is_bucket=True,
                    bucket_info=feature.BucketInfo(
                        method="bucket_attribute_specified_bucket_value",
                        number=4,
                        setting=(),
                    ),
                    require_training_set=True,
                ),
                "ref_fre_rank": feature.Value(
                    dtype="float",
                    description=(
                        "average training-set frequency rank of words in sentence"
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
                                description=("capitalness of token"),
                                is_bucket=True,
                                bucket_info=feature.BucketInfo(
                                    method="bucket_attribute_discrete_value",
                                    number=4,
                                    setting=1,
                                ),
                            ),
                            "tok_position": feature.Value(
                                dtype="float",
                                description=("relative position of token in sentence"),
                                is_bucket=True,
                                bucket_info=feature.BucketInfo(
                                    method="bucket_attribute_specified_bucket_value",
                                    number=4,
                                    setting=(),
                                ),
                            ),
                            "tok_chars": feature.Value(
                                dtype="float",
                                description="number of characters in the token",
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
    def default_metrics(
        cls, source_language=None, target_language=None
    ) -> list[MetricConfig]:
        defaults = ['rouge1', 'rouge2', 'rougeL', 'bleu', 'length_ratio']
        return [
            EaaSMetricConfig(
                name=x, source_language=source_language, target_language=target_language
            )
            for x in defaults
        ]

    @classmethod
    def full_metric_list(
        cls, source_language=None, target_language=None
    ) -> list[MetricConfig]:
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
            "length",
            "length_ratio",
        ]
        return [
            EaaSMetricConfig(
                name=x, source_language=source_language, target_language=target_language
            )
            for x in full_metrics
        ]

    # --- Feature functions accessible by ExplainaboardBuilder._get_feature_func()
    def _get_source_length(self, sys_info: SysOutputInfo, existing_features: dict):
        return len(unwrap(sys_info.source_tokenizer)(existing_features["source"]))

    def _get_reference_length(self, sys_info: SysOutputInfo, existing_features: dict):
        return len(unwrap(sys_info.target_tokenizer)(existing_features["reference"]))

    def _get_hypothesis_length(self, sys_info: SysOutputInfo, existing_features: dict):
        return len(unwrap(sys_info.target_tokenizer)(existing_features["hypothesis"]))

    # training set dependent features (could be merged for optimization?)
    def _get_src_num_oov(
        self, sys_info: SysOutputInfo, existing_features: dict, statistics: Any
    ):
        return explainaboard.utils.feature_funcs.feat_num_oov(
            existing_features,
            statistics['source_vocab'],
            lambda x: x['source'],
            unwrap(sys_info.source_tokenizer),
        )

    def _get_src_fre_rank(
        self, sys_info: SysOutputInfo, existing_features: dict, statistics: Any
    ):
        return explainaboard.utils.feature_funcs.feat_freq_rank(
            existing_features,
            statistics['source_vocab'],
            lambda x: x['source'],
            unwrap(sys_info.source_tokenizer),
        )

    def _get_ref_num_oov(
        self, sys_info: SysOutputInfo, existing_features: dict, statistics: Any
    ):
        return explainaboard.utils.feature_funcs.feat_num_oov(
            existing_features,
            statistics['target_vocab'],
            lambda x: x['reference'],
            unwrap(sys_info.target_tokenizer),
        )

    def _get_ref_fre_rank(
        self, sys_info: SysOutputInfo, existing_features: dict, statistics: Any
    ):
        return explainaboard.utils.feature_funcs.feat_freq_rank(
            existing_features,
            statistics['target_vocab'],
            lambda x: x['reference'],
            unwrap(sys_info.target_tokenizer),
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
        if name in self._default_features:
            return self._default_features[name]
        else:
            return self._default_features['ref_tok_info'][name]

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
            for ref_tok in unwrap(sys_info.target_tokenizer)(dict_sysout['reference']):
                ref_test_freq[ref_tok] = ref_test_freq.get(ref_tok, 0) + 1
            for src_tok in unwrap(sys_info.source_tokenizer)(dict_sysout['source']):
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
                feature_info = sys_features[bucket_feature]
                feature_func = self._get_feature_func(
                    bucket_feature, feature_info.is_custom
                )
                bucket_feature_funcs[bucket_feature] = (
                    feature_func,
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
            ref_toks = unwrap(sys_info.target_tokenizer)(dict_sysout['reference'])
            hyp_toks = unwrap(sys_info.target_tokenizer)(dict_sysout['hypothesis'])
            dict_sysout["ref_tok_info"] = self._complete_tok_features(
                ref_toks, hyp_toks, ref_test_freq, statistics=external_stats
            )
            dict_sysout["hyp_tok_info"] = self._complete_tok_features(
                hyp_toks, ref_toks, ref_test_freq, statistics=external_stats
            )

        return active_features

    def _complete_tok_features(
        self, toks: TokenSeq, other_toks: TokenSeq, ref_test_freq, statistics=None
    ):

        # Get training set stats if they exist
        has_stats = statistics is not None and len(statistics) > 0
        fre_dic = statistics["target_vocab"] if has_stats else None

        # Find tokens in other set
        other_tok_list = defaultdict(list)
        for i, tok in enumerate(other_toks):
            other_tok_list[tok].append(i)

        tok_dics = []
        for i, tok in enumerate(toks):
            # Basic features
            my_other = other_tok_list.get(tok, list())
            matched = my_other.pop(0) if len(my_other) > 0 else -1
            start = toks.positions[i]
            tok_dic = {
                'tok_text': tok,
                'tok_pos': (i, i + 1),
                'tok_char_pos': (start, start + len(tok)),
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

    def _get_sample_features(
        self, sys_output: list[dict], feats: list[str]
    ) -> list[tuple[BucketCase, list]]:
        sample_features: list[tuple[BucketCase, list]] = []
        for samp_id, my_output in enumerate(sys_output):
            # Get reference-only and matched cases
            for ref_id, ref_info in enumerate(my_output['ref_tok_info']):
                ref_span = BucketCaseSpan(
                    sample_id=samp_id,
                    token_span=ref_info['tok_pos'],
                    char_span=ref_info['tok_char_pos'],
                    orig_str='reference',
                    text=ref_info['tok_text'],
                )
                ref_feats = [ref_info[x] for x in feats]
                if ref_info['tok_matched'] < 0:
                    sample_features.append((ref_span, ref_feats))
                else:
                    hyp_info = my_output['hyp_tok_info'][ref_info['tok_matched']]
                    hyp_span = BucketCaseSpan(
                        sample_id=samp_id,
                        token_span=hyp_info['tok_pos'],
                        char_span=hyp_info['tok_char_pos'],
                        orig_str='hypothesis',
                        text=hyp_info['tok_text'],
                    )
                    both_span = BucketCaseMultiSpan(
                        sample_id=samp_id, spans=[ref_span, hyp_span]
                    )
                    sample_features.append((both_span, ref_feats))
            for hyp_id, hyp_info in enumerate(my_output['hyp_tok_info']):
                if hyp_info['tok_matched'] < 0:
                    hyp_span = BucketCaseSpan(
                        sample_id=samp_id,
                        token_span=hyp_info['tok_pos'],
                        char_span=hyp_info['tok_char_pos'],
                        orig_str='reference',
                        text=hyp_info['tok_text'],
                    )
                    hyp_feats = [hyp_info[x] for x in feats]
                    sample_features.append((hyp_span, hyp_feats))
        return sample_features

    def bucketing_samples(
        self,
        sys_info: SysOutputInfo,
        sys_output: list[dict],
        active_features: list[str],
        metric_stats: list[MetricStats],
    ) -> dict[str, list[BucketPerformance]]:

        features = unwrap(sys_info.features)
        sent_feats: list[str] = []
        tok_feats: list[str] = []
        for x in active_features:
            (sent_feats if (x in features) else tok_feats).append(x)

        # First, get the buckets for sentences using the standard protocol
        performances_over_bucket = super().bucketing_samples(
            sys_info, sys_output, sent_feats, metric_stats
        )

        all_sample_features = self._get_sample_features(sys_output, tok_feats)

        # Second, get the buckets for tokens
        for feature_id, feature_name in enumerate(
            progress(tok_feats, desc="bucketing token features")
        ):

            # Choose behavior based on whether this is a feature of samples or spans
            my_feature = features["ref_tok_info"].feature.feature[feature_name]
            bucket_info = my_feature.bucket_info

            # Get buckets for true spans
            bucket_func: Callable[..., list[BucketCaseCollection]] = getattr(
                bucketing, bucket_info.method
            )

            sample_features = [
                (case, feats[feature_id]) for case, feats in all_sample_features
            ]

            samples_over_bucket = bucket_func(
                sample_features=sample_features,
                bucket_number=bucket_info.number,
                bucket_setting=bucket_info.setting,
            )

            # evaluating bucket: get bucket performance
            performances_over_bucket[feature_name] = self.get_bucket_performance_tok(
                sys_info,
                sys_output,
                samples_over_bucket,
            )

        return performances_over_bucket

    def get_bucket_performance_tok(
        self,
        sys_info: SysOutputInfo,
        sys_output: list[dict],
        samples_over_bucket: list[BucketCaseCollection],
    ) -> list[BucketPerformance]:
        """
        This function defines how to get bucket-level performance w.r.t a given feature
        (e.g., sentence length)
        :param sys_info: Information about the system output
        :param sys_output: The system output itself
        :param samples_over_bucket: a list of bucket collections
        :return: bucket_name_to_performance: a dictionary that maps bucket names to
            bucket performance
        """

        bucket_performances: list[BucketPerformance] = []
        f1_score = F1ScoreConfig(name='F1').to_metric()
        for bucket_collection in samples_over_bucket:
            stats_list = []
            for case in bucket_collection.samples:
                # Both ref and hyp exist, so matched
                if isinstance(case, BucketCaseMultiSpan):
                    stats_list.append([1.0, 1.0, 1.0])
                elif cast(BucketCaseSpan, case).orig_str == 'reference':
                    stats_list.append([1.0, 0.0, 0.0])
                else:
                    stats_list.append([0.0, 1.0, 0.0])
            bucket_samples = self._subsample_bucket_cases(bucket_collection.samples)
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
                bucket_interval=bucket_collection.interval,
                n_samples=len(bucket_collection.samples),
                bucket_samples=bucket_samples,
                performances=[performance],
            )
            bucket_performances.append(bucket_performance)

        return bucket_performances

    @aggregating()
    def _statistics_func(self, samples: Iterator, sys_info: SysOutputInfo):
        return {
            'source_vocab': accumulate_vocab_from_samples(
                samples, lambda x: x['source'], unwrap(sys_info.source_tokenizer)
            ),
            'target_vocab': accumulate_vocab_from_samples(
                samples, lambda x: x['reference'], unwrap(sys_info.target_tokenizer)
            ),
        }
