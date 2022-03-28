from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast, Optional

import numpy
from tqdm import tqdm

from explainaboard import feature
from explainaboard.info import BucketPerformance, Performance, SysOutputInfo
from explainaboard.processors.processor import Processor
from explainaboard.processors.processor_registry import register_processor
from explainaboard.tasks import TaskType
from explainaboard.utils.analysis import cap_feature
import explainaboard.utils.bucketing
import explainaboard.utils.feature_funcs
from explainaboard.utils.py_utils import sort_dict
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
                    description="the average rank of each word in the source sentence based on its frequency in training "
                    "set",
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
                    feature.Set(
                        {
                            "tok_text": feature.Value("string"),
                            "tok_pos": feature.Position(positions=[0, 0]),
                            "tok_matched": feature.Value(
                                dtype="bool",
                                description="whether the ref/hyp token matches with a hyp/ref token",
                                is_bucket=False,
                            ),
                            "tok_capitalness": feature.Value(
                                dtype="string",
                                description="The capitalness of an token. For example, first_caps represents only the "
                                "first character of the token is capital. full_caps denotes all characters "
                                "of the token are capital",
                                is_bucket=True,
                                bucket_info=feature.BucketInfo(
                                    method="bucket_attribute_discrete_value",
                                    number=4,
                                    setting=1,
                                ),
                            ),
                            "tok_position": feature.Value(
                                dtype="float",
                                description="The relative position of a token in a sentence",
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
    def default_metrics(cls) -> list[str]:
        return ["rouge1", "rouge2", "rougeL", "bleu"]

    # --- Feature functions accessible by ExplainaboardBuilder._get_feature_func()
    def _get_source_length(self, existing_features: dict):
        return len(self._tokenizer(existing_features["source"]))

    def _get_reference_length(self, existing_features: dict):
        return len(self._tokenizer(existing_features["reference"]))

    def _get_hypothesis_length(self, existing_features: dict):
        return len(self._tokenizer(existing_features["hypothesis"]))

    # training set dependent features (could be merged for optimization?)
    def _get_num_oov(self, existing_features: dict, statistics: Any):
        return explainaboard.utils.feature_funcs.feat_num_oov(
            existing_features, statistics, lambda x: x['source'], self._tokenizer
        )

    def _get_src_fre_rank(self, existing_features: dict, statistics: Any):
        return explainaboard.utils.feature_funcs.feat_freq_rank(
            existing_features, statistics, lambda x: x['source'], self._tokenizer
        )

    def _gen_scoring_stats(
        self, sys_info: SysOutputInfo, sys_output: list[dict]
    ) -> dict[str, str]:
        """Generate sufficient statistics for scoring.

        :param sys_info: Information about the system outputs
        :param sys_output: The system output itself
        :return: Statistics sufficient for scoring
        """
        inputs = []
        for _id, feature_table in enumerate(sys_output):
            inputs.append(
                {
                    "source": feature_table["source"],
                    "references": [feature_table["reference"]],
                    "hypothesis": feature_table["hypothesis"],
                }
            )
            sys_output[_id] = feature_table

        request_id = self._get_eaas_client().async_score(
            inputs,
            task="sum",  # TODO(pengfei): this should be generalized
            metrics=unwrap(sys_info.metric_names).copy(),
            lang="en",
            cal_attributes=False,
        )

        # Note that this returns an asynchronous request ID so the EaaS call can continue while other
        # faeturizing, etc. is going on
        return {'request_id': request_id}

    # TODO(odashi): Restructure this function (and EaaS client) to be type-safe.
    def _fetch_scoring_stats(self, scoring_stats: dict[str, Any]):
        """
        A utility function used to lazily fetch the actual scoring dict when it's necessary.
        """
        if 'request_id' in scoring_stats:
            eaas_stats: dict[str, Any] = unwrap(self._eaas_client).wait_and_get_result(
                scoring_stats['request_id']
            )
            scoring_stats.clear()
            for k, v in eaas_stats.items():
                scoring_stats[k] = v

    def _get_feature_info(self, name: str):
        if name in self._features:
            return self._features[name]
        else:
            return self._features['ref_tok_info'][name]

    def _complete_features(
        self, sys_info: SysOutputInfo, sys_output: list[dict], external_stats=None
    ) -> Optional[list[str]]:
        """
        This function is used to calculate features used for bucketing, such as sentence_length
        :return:
        """

        # One pass over the test set to find token test frequency
        ref_test_freq: dict[str, int] = {}
        src_test_freq: dict[str, int] = {}
        for dict_sysout in sys_output:
            for ref_tok in self._tokenizer(dict_sysout['reference']):
                ref_test_freq[ref_tok] = ref_test_freq.get(ref_tok, 0) + 1
            for src_tok in self._tokenizer(dict_sysout['source']):
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
                    dict_sysout[bucket_key] = bucket_func(dict_sysout, external_stats)
                else:
                    dict_sysout[bucket_key] = bucket_func(dict_sysout)
                    # print(dict_sysout[bucket_key])

            # span features for true and predicted spans
            ref_toks = self._tokenizer(dict_sysout['reference'])
            hyp_toks = self._tokenizer(dict_sysout['hypothesis'])
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
        other_tok_count = {}
        for tok in other_toks:
            other_tok_count[tok] = other_tok_count.get(tok, 0) + 1

        tok_dics = []
        for i, tok in enumerate(toks):
            # Basic features
            matched = other_tok_count.get(tok, 0) > 0
            if matched:
                other_tok_count[tok] -= 1
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

    def get_overall_performance(
        self,
        sys_info: SysOutputInfo,
        sys_output: list[dict],
        scoring_stats: Any = None,
    ) -> dict[str, Performance]:

        # Fetch asynchronously calculated stats
        self._fetch_scoring_stats(cast('dict[str, Any]', scoring_stats))

        overall = {}
        for metric_name in unwrap_generator(sys_info.metric_names):

            scoring_stats_typed = cast('dict[str, dict[str, float]]', scoring_stats)
            overall_value = scoring_stats_typed["corpus_level"]["corpus_" + metric_name]
            overall_performance = Performance(
                metric_name=metric_name,
                value=overall_value,
            )

            overall[metric_name] = overall_performance
        return overall

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
        scoring_stats: Any = None,
    ) -> tuple[dict, dict]:

        features = unwrap(sys_info.features)
        sent_feats: list[str] = []
        tok_feats: list[str] = []
        for x in active_features:
            (sent_feats if (x in features) else tok_feats).append(x)

        # First, get the buckets for sentences using the standard protocol
        samples_over_bucket, performances_over_bucket = super()._bucketing_samples(
            sys_info, sys_output, sent_feats, scoring_stats
        )
        samples_over_bucket_pred = {}

        # Second, get the buckets for tokens
        for feature_name in tqdm(tok_feats, desc="bucketing token features"):

            # Choose behavior based on whether this is a feature of samples or spans
            my_feature = features["ref_tok_info"].feature.feature[feature_name]
            bucket_info = my_feature.bucket_info

            # Get buckets for true spans
            bucket_func = getattr(explainaboard.utils.bucketing, bucket_info.method)

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
            ] = explainaboard.utils.bucketing.bucket_attribute_specified_bucket_interval(
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
        samples_over_bucket_true: dict[str, list[str]],
        samples_over_bucket_pred: dict[str, list[str]],
    ) -> dict[str, list[BucketPerformance]]:
        """
        This function defines how to get bucket-level performance w.r.t a given feature (e.g., sentence length)
        :param sys_info: Information about the system output
        :param sys_output: The system output itself
        :param samples_over_bucket_true: a dictionary mapping bucket interval names to true sample IDs
        :param samples_over_bucket_pred: a dictionary mapping bucket interval names to predicted sample IDs
        :return: bucket_name_to_performance: a dictionary that maps bucket names to bucket performance
        """

        bucket_name_to_performance: dict[str, list[BucketPerformance]] = {}
        for bucket_interval, toks_true in samples_over_bucket_true.items():

            if bucket_interval not in samples_over_bucket_pred.keys():
                raise ValueError("Predict Label Bucketing Errors")
            else:
                toks_pred = samples_over_bucket_pred[bucket_interval]

            p_denom, r_denom = len(toks_pred), len(toks_true)

            # TODO(odashi): I didn't understand what these lines are doing.
            # These lines clearly violate type hints as far as I believed those in
            # the argument list, but I couldn't figure out the correct typing.
            p_num = sum(
                map(
                    lambda x: sys_output[x[0]]['hyp_tok_info'][x[1]]['tok_matched'],
                    toks_pred,
                )
            )
            r_num = sum(
                map(
                    lambda x: sys_output[x[0]]['ref_tok_info'][x[1]]['tok_matched'],
                    toks_true,
                )
            )

            p = p_num / float(p_denom) if p_denom else 0.0
            r = r_num / float(r_denom) if r_denom else 0.0
            f1 = 2 * p * r / (p + r) if p + r else 0.0

            bucket_name_to_performance[bucket_interval] = []
            for metric_name, metric_value in [
                ('f1', f1),
                ('precision', p),
                ('recall', r),
            ]:
                bucket_performance = BucketPerformance(
                    bucket_name=bucket_interval,
                    metric_name=metric_name,
                    value=metric_value,
                    n_samples=len(toks_true),
                    bucket_samples=toks_true,
                )
                bucket_name_to_performance[bucket_interval].append(bucket_performance)

        return sort_dict(bucket_name_to_performance)

    def get_bucket_performance(
        self,
        sys_info: SysOutputInfo,
        sys_output: list[dict],
        samples_over_bucket: dict[str, list[int]],
        scoring_stats: Any = None,
    ) -> dict[str, list[BucketPerformance]]:
        """
        This function defines how to get bucket-level performance w.r.t a given feature (e.g., sentence length)
        :param sys_info: Information about the system output
        :param sys_output: The system output itself
        :param samples_over_bucket: a dictionary mapping bucket interval names to sample IDs for that bucket
        :return: bucket_name_to_performance: a dictionary that maps bucket names to bucket performance
        """

        # Fetch asynchronously calculated stats
        self._fetch_scoring_stats(cast('dict[str, Any]', scoring_stats))

        bucket_name_to_performance: dict[str, list[BucketPerformance]] = {}
        for bucket_interval, sample_ids in samples_over_bucket.items():
            bucket_cases = []
            bucket_inputs = []
            dict_metric_to_values: dict[str, list[float]] = {}

            for sample_id in sample_ids:
                sys_out = sys_output[sample_id]
                bucket_inputs.append(
                    {
                        "source": sys_out["source"],
                        "references": [sys_out["reference"]],
                        "hypothesis": sys_out["hypothesis"],
                    }
                )

                if sys_info.is_print_case:
                    bucket_case = str(sample_id)
                    bucket_cases.append(bucket_case)

                # TODO(gneubig): This needs to be fixed because many metrics are not linearly decomposable
                for metric_name in unwrap(sys_info.metric_names):
                    scoring_stats_typed = cast(
                        'dict[str, dict[int, dict[str, float]]]', scoring_stats
                    )
                    metric_value = scoring_stats_typed["sample_level"][int(sample_id)][
                        metric_name
                    ]  # This would be modified later
                    if metric_name not in dict_metric_to_values.keys():
                        dict_metric_to_values[metric_name] = [metric_value]
                    else:
                        dict_metric_to_values[metric_name].append(metric_value)

            bucket_name_to_performance[bucket_interval] = []

            for metric_name in unwrap_generator(sys_info.metric_names):

                bucket_value = numpy.average(dict_metric_to_values[metric_name])

                bucket_performance = BucketPerformance(
                    bucket_name=bucket_interval,
                    metric_name=metric_name,
                    value=bucket_value,
                    n_samples=len(dict_metric_to_values[metric_name]),
                    bucket_samples=bucket_cases,
                )

                bucket_name_to_performance[bucket_interval].append(bucket_performance)

        return sort_dict(bucket_name_to_performance)
