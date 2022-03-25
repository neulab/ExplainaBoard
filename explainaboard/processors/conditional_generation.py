from typing import Any, Optional, Tuple, Callable
from typing import Iterator, Dict, List

import numpy
from tqdm import tqdm


from datalabs import aggregating
import explainaboard.utils.feature_funcs
from build.lib.explainaboard.utils.analysis import cap_feature
from explainaboard import feature
from explainaboard.info import SysOutputInfo, Performance, BucketPerformance
from explainaboard.processors.processor import Processor
from explainaboard.processors.processor_registry import register_processor
from explainaboard.tasks import TaskType
from explainaboard.utils.py_utils import sort_dict
from explainaboard.utils.tokenizer import SingleSpaceTokenizer
import explainaboard.utils.bucketing


@register_processor(TaskType.conditional_generation)
class ConditionalGenerationProcessor(Processor):
    _task_type = TaskType.conditional_generation
    _default_metrics = ["rouge1", "rouge2", "rougeL", "bleu"]

    _features = feature.Features(
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
                            description="tok test frequency in the training set",
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
                            description="tok test frequency in the training set",
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

    def __init__(self):
        super().__init__()
        # self._statistics_func = get_statistics

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
        self, sys_info: SysOutputInfo, sys_output: List[dict]
    ) -> Any:
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
            metrics=sys_info.metric_names.copy(),
            lang="en",
            cal_attributes=False,
        )

        # Note that this returns an asynchronous request ID so the EaaS call can continue while other
        # faeturizing, etc. is going on
        return {'request_id': request_id}

    def _fetch_scoring_stats(self, scoring_stats: Any):
        """
        A utility function used to lazily fetch the actual scoring dict when it's necessary.
        """
        if 'request_id' in scoring_stats:
            eaas_stats = self._eaas_client.wait_and_get_result(
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
        self, sys_info: SysOutputInfo, sys_output: List[dict], external_stats=None
    ) -> Optional[List[str]]:
        """
        This function is used to calculate features used for bucketing, such as sentence_length
        :return:
        """

        # One pass over the test set to find token test frequency
        ref_test_freq, src_test_freq = {}, {}
        for dict_sysout in sys_output:
            for ref_tok in self._tokenizer(dict_sysout['reference']):
                ref_test_freq[ref_tok] = ref_test_freq.get(ref_tok, 0) + 1
            for src_tok in self._tokenizer(dict_sysout['source']):
                src_test_freq[src_tok] = src_test_freq.get(src_tok, 0) + 1

        # Get names of bucketing features
        bucket_feature_funcs = {}
        active_features = list(
            sys_info.features.get_bucket_features(
                include_training_dependent=external_stats is not None
            )
        )
        for bucket_feature in active_features:
            if bucket_feature in sys_info.features:
                bucket_feature_funcs[bucket_feature] = (
                    self._get_feature_func(bucket_feature),
                    sys_info.features[bucket_feature].require_training_set,
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
        sys_output: List[dict],
        scoring_stats: Any = None,
    ) -> Dict[str, Performance]:

        # Fetch asynchronously calculated stats
        self._fetch_scoring_stats(scoring_stats)

        overall = {}
        for metric_name in sys_info.metric_names:

            overall_value = scoring_stats["corpus_level"]["corpus_" + metric_name]
            overall_performance = Performance(
                metric_name=metric_name,
                value=overall_value,
            )

            overall[metric_name] = overall_performance
        return overall

    def _get_feature_dict(
        self, sys_output: List[dict], feature_name: str, output_to_toks: Callable
    ):
        feat_dict = {}
        for samp_id, my_output in enumerate(sys_output):
            for tok_id, tok_info in enumerate(output_to_toks(my_output)):
                feat_dict[(samp_id, tok_id)] = tok_info[feature_name]
        return feat_dict

    def _bucketing_samples(
        self,
        sys_info: SysOutputInfo,
        sys_output: List[dict],
        active_features: List[str],
        scoring_stats: Any = None,
    ) -> Tuple[dict, dict]:

        features = sys_info.features
        sent_feats, tok_feats = [], []
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
        sys_output: List[dict],
        samples_over_bucket_true: Dict[str, List[str]],
        samples_over_bucket_pred: Dict[str, List[str]],
    ) -> Dict[str, List[BucketPerformance]]:
        """
        This function defines how to get bucket-level performance w.r.t a given feature (e.g., sentence length)
        :param sys_info: Information about the system output
        :param sys_output: The system output itself
        :param samples_over_bucket_true: a dictionary mapping bucket interval names to true sample IDs
        :param samples_over_bucket_pred: a dictionary mapping bucket interval names to predicted sample IDs
        :return: bucket_name_to_performance: a dictionary that maps bucket names to bucket performance
        """

        bucket_name_to_performance = {}
        for bucket_interval, toks_true in samples_over_bucket_true.items():

            if bucket_interval not in samples_over_bucket_pred.keys():
                raise ValueError("Predict Label Bucketing Errors")
            else:
                toks_pred = samples_over_bucket_pred[bucket_interval]

            p_denom, r_denom = len(toks_pred), len(toks_true)
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
        sys_output: List[dict],
        samples_over_bucket: Dict[str, List[int]],
        scoring_stats: Any = None,
    ) -> Dict[str, List[BucketPerformance]]:
        """
        This function defines how to get bucket-level performance w.r.t a given feature (e.g., sentence length)
        :param sys_info: Information about the system output
        :param sys_output: The system output itself
        :param samples_over_bucket: a dictionary mapping bucket interval names to sample IDs for that bucket
        :return: bucket_name_to_performance: a dictionary that maps bucket names to bucket performance
        """

        # Fetch asynchronously calculated stats
        self._fetch_scoring_stats(scoring_stats)

        bucket_name_to_performance = {}
        for bucket_interval, sample_ids in samples_over_bucket.items():

            bucket_cases = []

            bucket_inputs = []
            dict_metric_to_values = {}

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
                for metric_name in sys_info.metric_names:
                    metric_value = scoring_stats["sample_level"][int(sample_id)][
                        metric_name
                    ]  # This would be modified later
                    if metric_name not in dict_metric_to_values.keys():
                        dict_metric_to_values[metric_name] = [metric_value]
                    else:
                        dict_metric_to_values[metric_name].append(metric_value)

            bucket_name_to_performance[bucket_interval] = []

            for metric_name in sys_info.metric_names:

                bucket_value = numpy.average(dict_metric_to_values[metric_name])

                # print(
                #       f"value:\t {bucket_value}\n"
                #       f"confidence low\t {confidence_score_low}\n"
                #       f"confidence up \t {confidence_score_high}\n"
                #       f"---------------------------------")

                bucket_performance = BucketPerformance(
                    bucket_name=bucket_interval,
                    metric_name=metric_name,
                    value=bucket_value,
                    n_samples=len(dict_metric_to_values[metric_name]),
                    bucket_samples=bucket_cases,
                )

                bucket_name_to_performance[bucket_interval].append(bucket_performance)

        return sort_dict(bucket_name_to_performance)


# @register_processor(TaskType.summarization)
# class SummarizationProcessor(ConditionalGenerationProcessor):
#     _task_type = TaskType.summarization
#     _default_metrics = ["rouge1", "rouge2", "rougeL"]


# TODO(gneubig) should be conditional generation, not summarization
# Aggregate training set statistics
@aggregating(
    name="get_statistics",
    contributor="datalab",
    task="summarization",
    description="Calculate the overall statistics (e.g., density) of a given summarization dataset",
)
def get_statistics(samples: Iterator):
    """
    Input:
    samples: [{
     "text":
     "summary":
    }]
    Output:dict:
    """

    # TODO(gneubig): BEWARE THIS IS HACKY. This should use the same tokenizer as the processor.
    tokenizer = SingleSpaceTokenizer()

    vocab = {}
    vocab_pruning = {}

    for sample in tqdm(samples):

        text, summary = sample["text"], sample["summary"]

        # # oracle_position_fre
        # oracle_info = get_oracle_summary.func(sample)
        # index_of_oracles = [
        #     i for i, e in enumerate(oracle_info["oracle_labels"]) if e != 0
        # ]
        # oracle_position = str(int(numpy.mean(index_of_oracles)))
        #
        # if oracle_position not in oracle_position_fre.keys():
        #     oracle_position_fre[oracle_position] = 1
        # else:
        #     oracle_position_fre[oracle_position] += 1

        # Vocabulary info
        for w in tokenizer(text + summary):
            if w in vocab.keys():
                vocab[w] += 1
            else:
                vocab[w] = 1

    for k, v in vocab.items():  # pruning for the availability of database storage
        if v > 20:
            vocab_pruning[k] = v
        if len(vocab_pruning) > 100:
            break

    # the rank of each word based on its frequency
    sorted_dict = {
        key: rank
        for rank, key in enumerate(sorted(set(vocab_pruning.values()), reverse=True), 1)
    }
    vocab_rank = {k: sorted_dict[v] for k, v in vocab_pruning.items()}

    return {
        "vocab": vocab_pruning,
        "vocab_rank": vocab_rank,
        # "oracle_position_fre": oracle_position_fre,
    }
