from __future__ import annotations

import abc
from collections.abc import Callable
import copy
from typing import cast

from datalabs import aggregating, Dataset

from explainaboard import feature
from explainaboard.info import (
    BucketCaseCollection,
    BucketCaseLabeledSpan,
    BucketPerformance,
    Performance,
    SysOutputInfo,
)
from explainaboard.loaders.file_loader import DatalabFileLoader
from explainaboard.metric import F1ScoreConfig, MetricStats
from explainaboard.processors.processor import Processor
from explainaboard.utils import bucketing
from explainaboard.utils.logging import progress
from explainaboard.utils.span_utils import Span, SpanOps
from explainaboard.utils.typing_utils import unwrap


class SeqLabProcessor(Processor):
    @classmethod
    @abc.abstractmethod
    def default_span_ops(cls) -> SpanOps:
        """Returns the default metrics of this processor."""
        ...

    _DEFAULT_TAG = 'O'

    def __init__(self):
        super().__init__()
        self._span_ops: SpanOps = self.default_span_ops()

    @classmethod
    def default_features(cls) -> feature.Features:
        return feature.Features(
            {
                "tokens": feature.Sequence(feature=feature.Value("string")),
                "true_tags": feature.Sequence(feature=feature.Value("string")),
                "pred_tags": feature.Sequence(feature=feature.Value("string")),
                # --- the following are features of the sentences ---
                "sentence_length": feature.Value(
                    dtype="float",
                    description="sentence length",
                    is_bucket=True,
                    bucket_info=feature.BucketInfo(
                        method="bucket_attribute_specified_bucket_value",
                        number=4,
                        setting=(),
                    ),
                ),
                "span_density": feature.Value(
                    dtype="float",
                    description="the ration between all entity "
                    "tokens and sentence tokens ",
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
                        "the average rank of each word based on its frequency in "
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
                # --- the following are features of each entity ---
                "true_span_info": feature.Sequence(
                    feature=feature.Dict(
                        feature={
                            "span_text": feature.Value("string"),
                            "span_tokens": feature.Value(
                                dtype="float",
                                description="entity length",
                                is_bucket=True,
                                bucket_info=feature.BucketInfo(
                                    method="bucket_attribute_specified_bucket_value",
                                    number=4,
                                    setting=(),
                                ),
                            ),
                            "span_pos": feature.Position(positions=[0, 0]),
                            "span_tag": feature.Value(
                                dtype="string",
                                description="entity tag",
                                is_bucket=True,
                                bucket_info=feature.BucketInfo(
                                    method="bucket_attribute_discrete_value",
                                    number=4,
                                    setting=1,
                                ),
                            ),
                            "span_capitalness": feature.Value(
                                dtype="string",
                                description=(
                                    "The capitalness of an entity. For example, "
                                    "first_caps represents only the first character of "
                                    "the entity is capital. full_caps denotes all "
                                    "characters of the entity are capital"
                                ),
                                is_bucket=True,
                                bucket_info=feature.BucketInfo(
                                    method="bucket_attribute_discrete_value",
                                    number=4,
                                    setting=1,
                                ),
                            ),
                            "span_rel_pos": feature.Value(
                                dtype="float",
                                description=(
                                    "The relative position of an entity in a sentence"
                                ),
                                is_bucket=True,
                                bucket_info=feature.BucketInfo(
                                    method="bucket_attribute_specified_bucket_value",
                                    number=4,
                                    setting=(),
                                ),
                            ),
                            "span_chars": feature.Value(
                                dtype="float",
                                description="The number of characters of an entity",
                                is_bucket=True,
                                bucket_info=feature.BucketInfo(
                                    method="bucket_attribute_specified_bucket_value",
                                    number=4,
                                    setting=(),
                                ),
                            ),
                            "span_econ": feature.Value(
                                dtype="float",
                                description="entity label consistency",
                                is_bucket=True,
                                require_training_set=True,
                                bucket_info=feature.BucketInfo(
                                    method="bucket_attribute_specified_bucket_value",
                                    number=4,
                                    setting=(),
                                ),
                            ),
                            "span_efre": feature.Value(
                                dtype="float",
                                description="entity frequency",
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

    def _get_true_label(self, data_point: dict):
        return data_point["true_tags"]

    def _get_predicted_label(self, data_point: dict):
        return data_point["pred_tags"]

    @aggregating()
    def _statistics_func(self, samples: Dataset, sys_info: SysOutputInfo):
        dl_features = samples.info.features

        tokens_sequences = []
        tags_sequences = []

        vocab: dict[str, int] = {}
        tag_vocab: dict[str, int] = {}
        for sample in progress(samples):
            rep_sample = DatalabFileLoader.replace_labels(dl_features, sample)
            tokens, tags = rep_sample["tokens"], rep_sample["tags"]

            # update vocabulary
            for token, tag in zip(tokens, tags):
                vocab[token] = vocab.get(token, 0) + 1
                tag_vocab[tag] = tag_vocab.get(tag, 0) + 1

            tokens_sequences += tokens
            tags_sequences += tags

        # econ and efre dictionaries
        econ_dic, efre_dic = self.get_econ_efre_dic(tokens_sequences, tags_sequences)
        # vocab_rank: the rank of each word based on its frequency
        sorted_dict = {
            key: rank
            for rank, key in enumerate(sorted(set(vocab.values()), reverse=True), 1)
        }
        vocab_rank = {k: sorted_dict[v] for k, v in vocab.items()}

        return {
            "efre_dic": efre_dic,
            "econ_dic": econ_dic,
            "vocab": vocab,
            "vocab_rank": vocab_rank,
        }

    def _get_stat_values(
        self, econ_dic: dict, efre_dic: dict, span_text: str, span_tag: str
    ):
        """
        Get entity consistency and frequency values
        """
        span_tag = span_tag.lower()
        span_text = span_text.lower()
        econ_val = econ_dic.get(f'{span_text}|||{span_tag}', 0.0)
        efre_val = efre_dic.get(span_text, 0.0)
        return econ_val, efre_val

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

    def _complete_span_features(
        self, sentence, true_tags, pred_tags, statistics=None
    ) -> list[Span]:
        # Get training set stats if they exist
        has_stats = statistics is not None and len(statistics) > 0
        econ_dic = statistics["econ_dic"] if has_stats else None
        efre_dic = statistics["efre_dic"] if has_stats else None

        self._span_ops.set_resources(
            resources={
                "has_stats": has_stats,
                "econ_dic": econ_dic,
                "efre_dic": efre_dic,
            }
        )

        # Merge the spans together so that the span tag is "true_tag pred_tag"
        # using "_DEFAULT_TAG" if that span doesn't exist in the true or predicted tags
        # respectively
        # TODO(gneubig): This is probably calculating features twice, could be just once
        true_spans = self._span_ops.get_spans(toks=sentence, tags=true_tags)
        pred_spans = self._span_ops.get_spans(toks=sentence, tags=pred_tags)
        merged_spans = {}
        for span in true_spans:
            span.span_tag = f'{span.span_tag} {self._DEFAULT_TAG}'
            merged_spans[span.span_pos] = span
        for span in pred_spans:
            merged_span = merged_spans.get(span.span_pos)
            if not merged_span:
                span.span_tag = f'{self._DEFAULT_TAG} {span.span_tag}'
                merged_spans[span.span_pos] = span
            else:
                true_tag, _ = unwrap(merged_span.span_tag).split(' ')
                merged_span.span_tag = f'{true_tag} {span.span_tag}'

        return list(merged_spans.values())

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

        sent_feats: list[str] = []
        tok_feats: list[str] = []
        for x in active_features:
            (sent_feats if (x in sys_features) else tok_feats).append(x)

        for _id, dict_sysout in progress(enumerate(sys_output), desc="featurizing"):
            # Get values of bucketing features
            tokens = dict_sysout["tokens"]

            # sentence_length
            dict_sysout["sentence_length"] = len(tokens)

            # entity density
            dict_sysout["span_density"] = len(
                self._span_ops.get_spans_simple(tags=dict_sysout["true_tags"])
            ) / len(tokens)

            # sentence-level training set dependent features
            if external_stats is not None:
                dict_sysout["num_oov"] = self._get_num_oov(tokens, external_stats)
                dict_sysout["fre_rank"] = self._get_fre_rank(tokens, external_stats)

            # span features for true and predicted spans
            dict_sysout["span_info"] = self._complete_span_features(
                tokens,
                dict_sysout["true_tags"],
                dict_sysout["pred_tags"],
                statistics=external_stats,
            )

        # This is not used elsewhere, so just keep it as-is
        return active_features

    def bucketing_samples(
        self,
        sys_info: SysOutputInfo,
        sys_output: list[dict],
        active_features: list[str],
        metric_stats: list[MetricStats],
    ) -> dict[str, list[BucketPerformance]]:

        features = unwrap(sys_info.features)

        sent_feats: list[str] = []
        span_feats: list[str] = []
        for x in active_features:
            (sent_feats if (x in features) else span_feats).append(x)

        # First, get the buckets for sentences using the standard protocol
        performances_over_bucket = super().bucketing_samples(
            sys_info, sys_output, sent_feats, metric_stats
        )

        case_spans: list[tuple[BucketCaseLabeledSpan, Span]] = []
        for sample_id, my_output in enumerate(sys_output):
            for tok_id, span_info in enumerate(my_output['span_info']):
                span = cast(Span, span_info)
                true_tag, pred_tag = unwrap(span.span_tag).split(' ')
                case_spans.append(
                    (
                        BucketCaseLabeledSpan(
                            sample_id=sample_id,
                            token_span=unwrap(span.span_pos),
                            char_span=unwrap(span.span_char_pos),
                            orig_str='tokens',
                            text=unwrap(span.span_text),
                            true_label=true_tag,
                            predicted_label=pred_tag,
                        ),
                        span,
                    )
                )

        # Bucketing
        for feature_name in progress(span_feats, desc="span-level bucketing"):
            my_feature = features["true_span_info"].feature.feature[feature_name]
            bucket_info = my_feature.bucket_info

            # Get buckets for true spans
            bucket_func: Callable[..., list[BucketCaseCollection]] = getattr(
                bucketing, bucket_info.method
            )

            # Span tag is special because we keep track of both labels, keep just gold
            if feature_name == 'span_tag':
                sample_features = [
                    (case, unwrap(span.span_tag).split(' ')[0])
                    for case, span in case_spans
                ]
            else:
                sample_features = [
                    (case, getattr(span, feature_name)) for case, span in case_spans
                ]

            samples_over_bucket = bucket_func(
                sample_features=sample_features,
                bucket_number=bucket_info.number,
                bucket_setting=bucket_info.setting,
            )

            # evaluating bucket: get bucket performance
            performances_over_bucket[feature_name] = self.get_bucket_performance_seqlab(
                sys_info,
                sys_output,
                samples_over_bucket,
            )
        return performances_over_bucket

    def get_bucket_performance_seqlab(
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
        :param samples_over_bucket: a dictionary mapping bucket interval names to spans
        :return: bucket_name_to_performance: a dictionary that maps bucket names to
            bucket performance
        """
        bucket_metrics = [
            F1ScoreConfig(name='F1', ignore_classes=[self._DEFAULT_TAG]).to_metric()
        ]

        bucket_performances = []
        for bucket_collection in samples_over_bucket:

            # Get bucketed samples
            true_labels, pred_labels = [], []
            num_true = 0
            for sample in bucket_collection.samples:
                sample_span = cast(BucketCaseLabeledSpan, sample)
                true_labels.append(sample_span.true_label)
                num_true += 1 if sample_span.true_label != self._DEFAULT_TAG else 0
                pred_labels.append(sample_span.predicted_label)

            # Filter samples to error cases and limit number
            bucket_samples = []
            for x in bucket_collection.samples:
                bcls = cast(BucketCaseLabeledSpan, x)
                if bcls.true_label != bcls.predicted_label:
                    bucket_samples.append(bcls)
            bucket_samples = self._subsample_bucket_cases(bucket_samples)

            bucket_performance = BucketPerformance(
                bucket_interval=bucket_collection.interval,
                n_samples=num_true,
                bucket_samples=bucket_samples,
            )
            for metric in bucket_metrics:
                metric_val = metric.evaluate(
                    true_labels, pred_labels, conf_value=sys_info.conf_value
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

            bucket_performances.append(bucket_performance)

        return bucket_performances

    def get_econ_efre_dic(
        self, words: list[str], bio_tags: list[str]
    ) -> tuple[dict[str, float], dict[str, int]]:
        """
        Calculate the entity label consistency and frequency features from this paper
        https://aclanthology.org/2020.emnlp-main.489.pdf
        :param words: a list of all words in the corpus
        :param bio_tags: a list of all tags in the corpus
        :return: Returns two dictionaries:
                    econ: 'span|||tag' pointing to entity consistency values
                    efre: 'span' pointing to entity frequency values
        """
        chunks_train = self._span_ops.get_spans_simple(bio_tags)

        # Create pseudo-trie
        prefixes: set[str] = set()
        chunk_to_tag: dict[tuple[int, int], str] = {}
        entity_to_tagcnt: dict[str, dict[str, int]] = {}
        efre_dic: dict[str, int] = {}
        for true_chunk in progress(chunks_train):
            idx_start = true_chunk[1]
            idx_end = true_chunk[2]
            chunk_to_tag[(idx_start, idx_end)] = true_chunk[0]
            span_str = ''
            for i in range(0, idx_end - idx_start):
                w = words[idx_start + i].lower()
                span_str += w if i == 0 else f' {w}'
                prefixes.add(span_str)
            entity_to_tagcnt[span_str] = {}
            efre_dic[span_str] = efre_dic.get(span_str, 0) + 1

        # Actually calculate stats
        ltws = len(words)
        for idx_start in range(ltws):
            span_str = ''
            for i in range(0, ltws - idx_start):
                w = words[idx_start + i].lower()
                span_str += w if i == 0 else f' {w}'
                if span_str not in prefixes:
                    break
                if span_str in entity_to_tagcnt:
                    my_tag = chunk_to_tag.get(
                        (idx_start, idx_start + i + 1), self._DEFAULT_TAG
                    )
                    entity_to_tagcnt[span_str][my_tag] = (
                        entity_to_tagcnt[span_str].get(my_tag, 0) + 1
                    )

        econ_dic: dict[str, float] = {}
        for span_str, cnt_dic in entity_to_tagcnt.items():
            cnt_sum = float(sum(cnt_dic.values()))
            for tag, cnt in cnt_dic.items():
                econ_dic[f'{span_str}|||{tag}'] = cnt / cnt_sum
        return econ_dic, efre_dic

    def deserialize_system_output(self, output: dict) -> dict:
        new_output = copy.deepcopy(output)
        if "span_info" in new_output:
            new_output["span_info"] = [
                Span(**x) if isinstance(x, dict) else x for x in new_output["span_info"]
            ]
        return new_output
