"""A processor for the sequence labeling task."""

from __future__ import annotations

import abc
from collections.abc import Iterable
import copy
from typing import Any, cast

from explainaboard.analysis import feature
from explainaboard.analysis.analyses import (
    Analysis,
    AnalysisLevel,
    BucketAnalysis,
    ComboCountAnalysis,
)
from explainaboard.analysis.case import AnalysisCase, AnalysisCaseLabeledSpan
from explainaboard.analysis.feature import FeatureType
from explainaboard.analysis.feature_funcs import feat_freq_rank, feat_num_oov
from explainaboard.info import SysOutputInfo
from explainaboard.metrics.metric import MetricStats
from explainaboard.processors.processor import Processor
from explainaboard.utils.logging import progress
from explainaboard.utils.span_utils import cap_feature, Span, SpanOps
from explainaboard.utils.tokenizer import SingleSpaceTokenizer, Tokenizer
from explainaboard.utils.typing_utils import unwrap


class SeqLabProcessor(Processor):
    """A processor for the sequence labeling task."""

    @classmethod
    @abc.abstractmethod
    def _default_span_ops(cls) -> SpanOps:
        """Returns the default metrics of this processor."""
        ...

    _DEFAULT_TAG = "O"

    def __init__(self):
        """Constructor."""
        super().__init__()
        self._span_ops: SpanOps = self._default_span_ops()

    def get_tokenizer(self, lang: str | None) -> Tokenizer:
        """Get a tokenizer based on the language."""
        return SingleSpaceTokenizer()

    def default_analysis_levels(self) -> list[AnalysisLevel]:
        """See Processor.default_analysis_levels."""
        examp_features: dict[str, FeatureType] = {
            "tokens": feature.Sequence(
                feature=feature.Value(dtype=feature.DataType.STRING)
            ),
            "true_tags": feature.Sequence(
                feature=feature.Value(dtype=feature.DataType.STRING)
            ),
            "pred_tags": feature.Sequence(
                feature=feature.Value(dtype=feature.DataType.STRING)
            ),
            "text_length": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="text length in tokens",
                func=lambda info, x, c: len(x["tokens"]),
            ),
            "span_density": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="ratio of entity tokens to all tokens",
                func=lambda info, x, c: float(
                    len([y for y in x["true_tags"] if y != self._DEFAULT_TAG])
                )
                / len(x["true_tags"]),
            ),
            "num_oov": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="the number of out-of-vocabulary words",
                require_training_set=True,
                func=lambda info, x, c, stat: feat_num_oov(
                    info, x["tokens"], stat["vocab"], side="none"
                ),
            ),
            "fre_rank": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="average rank of each word based on training set frequency",
                require_training_set=True,
                func=lambda info, x, c, stat: feat_freq_rank(
                    info, x["tokens"], stat["vocab_rank"], side="none"
                ),
            ),
        }

        span_features: dict[str, FeatureType] = {
            "span_text": feature.Value(
                dtype=feature.DataType.STRING,
                description="text of the span",
                func=lambda info, x, c: c.text,
            ),
            "span_length": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="span length in tokens",
                func=lambda info, x, c: c.token_span[1] - c.token_span[0],
            ),
            "span_true_label": feature.Value(
                dtype=feature.DataType.STRING,
                description="true label of the span",
                func=lambda info, x, c: c.true_label,
            ),
            "span_pred_label": feature.Value(
                dtype=feature.DataType.STRING,
                description="predicted label of the span",
                func=lambda info, x, c: c.predicted_label,
            ),
            "span_capitalness": feature.Value(
                dtype=feature.DataType.STRING,
                description="whether the span is capitalized",
                func=lambda info, x, c: cap_feature(c.text),
            ),
            "span_rel_pos": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="relative position of the span",
                func=lambda info, x, c: c.token_span[0] / len(x["tokens"]),
            ),
            "span_chars": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="number of characters in the span",
                func=lambda info, x, c: len(c.text),
            ),
            "span_econ": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="consistency of the span labels",
                require_training_set=True,
                func=lambda info, x, c, stat: stat["econ_dic"].get(
                    f"{c.text.lower()}|||{c.true_label}", 0.0
                ),
            ),
            "span_efre": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="frequency of the span in the training set",
                require_training_set=True,
                func=lambda info, x, c, stat: stat["efre_dic"].get(c.text.lower(), 0.0),
            ),
        }

        return [
            AnalysisLevel(
                name="example",
                features=examp_features,
                metric_configs=self.default_metrics(level="example"),
            ),
            AnalysisLevel(
                name="span",
                features=span_features,
                metric_configs=self.default_metrics(level="span"),
            ),
        ]

    def default_analyses(self) -> list[Analysis]:
        """See Processor.default_analyses."""
        analysis_levels = self.default_analysis_levels()
        span_features = analysis_levels[1].features
        analyses: list[Analysis] = [
            BucketAnalysis(
                level="span",
                description=span_features["span_true_label"].description,
                feature="span_true_label",
                method="discrete",
                num_buckets=15,
            ),
            ComboCountAnalysis(
                level="span",
                description="confusion matrix",
                features=("span_true_label", "span_pred_label"),
            ),
            BucketAnalysis(
                level="span",
                description=span_features["span_capitalness"].description,
                feature="span_capitalness",
                method="discrete",
                num_buckets=4,
            ),
        ]
        analyses.extend(self.continuous_feature_analyses())
        return analyses

    def _get_true_label(self, data_point: dict):
        """See processor._get_true_label."""
        return data_point["true_tags"]

    def _get_predicted_label(self, data_point: dict):
        """See processor._get_predicted_label."""
        return data_point["pred_tags"]

    def _statistics_func(self, samples: Iterable[Any], sys_info: SysOutputInfo):

        tokens_sequences = []
        tags_sequences = []

        vocab: dict[str, int] = {}
        tag_vocab: dict[str, int] = {}
        for sample in progress(samples):
            tokens, tags = sample["tokens"], sample["true_tags"]

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

    def _gen_cases_and_stats(
        self,
        sys_info: SysOutputInfo,
        sys_output: list[dict],
        statistics: Any,
        analysis_level: AnalysisLevel,
    ) -> tuple[list[AnalysisCase], dict[str, MetricStats]]:
        if analysis_level.name == "example":
            return super()._gen_cases_and_stats(
                sys_info, sys_output, statistics, analysis_level
            )
        elif analysis_level.name != "span":
            raise ValueError(f"{analysis_level.name}-level analysis not supported")
        # Do span-level analysis
        cases: list[AnalysisCaseLabeledSpan] = []
        # Calculate features
        for i, output in progress(
            enumerate(sys_output), desc="calculating span-level features"
        ):
            # get the spans from each sentence
            tokens = output["tokens"]
            true_spans = self._span_ops.get_spans(toks=tokens, tags=output["true_tags"])
            pred_spans = self._span_ops.get_spans(toks=tokens, tags=output["pred_tags"])
            # merge the spans together
            merged_spans: dict[tuple[int, int], Span] = {}
            for span in true_spans:
                span.span_tag = f"{span.span_tag} {self._DEFAULT_TAG}"
                merged_spans[unwrap(span.span_pos)] = span
            for span in pred_spans:
                merged_span = merged_spans.get(unwrap(span.span_pos))
                if not merged_span:
                    span.span_tag = f"{self._DEFAULT_TAG} {span.span_tag}"
                    merged_spans[unwrap(span.span_pos)] = span
                else:
                    true_tag, _ = unwrap(merged_span.span_tag).split(" ")
                    merged_span.span_tag = f"{true_tag} {span.span_tag}"
            # analysis cases
            for ms in merged_spans.values():
                true_tag, pred_tag = unwrap(ms.span_tag).split(" ")
                case = AnalysisCaseLabeledSpan(
                    sample_id=i,
                    features={},
                    token_span=unwrap(ms.span_pos),
                    char_span=unwrap(ms.span_char_pos),
                    text=unwrap(ms.span_text),
                    true_label=true_tag,
                    predicted_label=pred_tag,
                    orig_str="source",
                )
                for feat_name, feat_spec in analysis_level.features.items():
                    if feat_spec.func is None:
                        raise ValueError(
                            f"could not find feature function for {feat_name}"
                        )
                    elif not feat_spec.require_training_set:
                        case.features[feat_name] = feat_spec.func(
                            sys_info, output, case
                        )
                    elif statistics is not None:
                        case.features[feat_name] = feat_spec.func(
                            sys_info, output, case, statistics
                        )
                cases.append(case)
        # calculate metric stats
        true_data = [x.true_label for x in cases]
        pred_data = [x.predicted_label for x in cases]
        metric_stats: dict[str, MetricStats] = {
            name: config.to_metric().calc_stats_from_data(true_data, pred_data)
            for name, config in analysis_level.metric_configs.items()
        }
        return cast(list[AnalysisCase], cases), metric_stats

    def get_econ_efre_dic(
        self, words: list[str], bio_tags: list[str]
    ) -> tuple[dict[str, float], dict[str, int]]:
        """Calculates entity label consistency and frequency features.

        Reference this paper:
        https://aclanthology.org/2020.emnlp-main.489.pdf

        Args:
            words: a list of all words in the corpus
            bio_tags: a list of all tags in the corpus

        Returns: two dictionaries:
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
            span_str = ""
            for i in range(0, idx_end - idx_start):
                w = words[idx_start + i].lower()
                span_str += w if i == 0 else f" {w}"
                prefixes.add(span_str)
            entity_to_tagcnt[span_str] = {}
            efre_dic[span_str] = efre_dic.get(span_str, 0) + 1

        # Actually calculate stats
        ltws = len(words)
        for idx_start in range(ltws):
            span_str = ""
            for i in range(0, ltws - idx_start):
                w = words[idx_start + i].lower()
                span_str += w if i == 0 else f" {w}"
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
                econ_dic[f"{span_str}|||{tag}"] = cnt / cnt_sum
        return econ_dic, efre_dic

    def deserialize_system_output(self, output: dict) -> dict:
        """See Processor.deserialize_system_output."""
        new_output = copy.deepcopy(output)
        if "span_info" in new_output:
            new_output["span_info"] = [
                Span(**x) if isinstance(x, dict) else x for x in new_output["span_info"]
            ]
        return new_output
