from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any, cast

import numpy as np

from explainaboard import TaskType
from explainaboard.analysis import feature
from explainaboard.analysis.analyses import Analysis, AnalysisLevel
from explainaboard.analysis.case import (
    AnalysisCase,
    AnalysisCaseMultiSpan,
    AnalysisCaseSpan,
)
from explainaboard.analysis.feature import FeatureType
from explainaboard.analysis.feature_funcs import (
    accumulate_vocab_from_samples,
    cap_feature,
    count_tokens,
    feat_freq_rank,
    feat_num_oov,
)
from explainaboard.info import SysOutputInfo
from explainaboard.metrics.eaas import (
    EaaSMetricConfig,
    EaaSMetricStats,
    get_eaas_client,
)
from explainaboard.metrics.external_eval import ExternalEvalConfig
from explainaboard.metrics.f1_score import F1ScoreConfig
from explainaboard.metrics.metric import MetricConfig, MetricStats, SimpleMetricStats
from explainaboard.processors.processor import Processor
from explainaboard.processors.processor_registry import register_processor
from explainaboard.utils.logging import progress
from explainaboard.utils.tokenizer import TokenSeq
from explainaboard.utils.typing_utils import unwrap, unwrap_generator


@register_processor(TaskType.conditional_generation)
class ConditionalGenerationProcessor(Processor):
    @classmethod
    def task_type(cls) -> TaskType:
        return TaskType.conditional_generation

    def default_analysis_levels(self) -> list[AnalysisLevel]:
        examp_features: dict[str, FeatureType] = {
            "source": feature.Value("string"),
            "reference": feature.Value("string"),
            "hypothesis": feature.Value("string"),
            "source_length": feature.Value(
                dtype="float",
                description="length of the source",
                func=lambda info, x, c: count_tokens(info, x['source'], side='source'),
            ),
            "reference_length": feature.Value(
                dtype="float",
                description="length of the reference",
                func=lambda info, x, c: count_tokens(
                    info, x['reference'], side='target'
                ),
            ),
            "hypothesis_length": feature.Value(
                dtype="float",
                description="length of the hypothesis",
                func=lambda info, x, c: count_tokens(
                    info, x['hypothesis'], side='target'
                ),
            ),
            "src_num_oov": feature.Value(
                dtype="float",
                description="OOV words in the source",
                func=lambda info, x, c, stat: feat_num_oov(
                    info, x['source'], stat['source_vocab'], side='source'
                ),
                require_training_set=True,
            ),
            "src_fre_rank": feature.Value(
                dtype="float",
                description="average training-set frequency rank of words in source",
                func=lambda info, x, c, stat: feat_freq_rank(
                    info, x['source'], stat['source_vocab_rank'], side='source'
                ),
                require_training_set=True,
            ),
            "ref_num_oov": feature.Value(
                dtype="float",
                description="number of OOV words in reference",
                func=lambda info, x, c, stat: feat_num_oov(
                    info, x['reference'], stat['target_vocab'], side='target'
                ),
                require_training_set=True,
            ),
            "ref_fre_rank": feature.Value(
                dtype="float",
                description=(
                    "average training-set frequency rank of words in reference"
                ),
                func=lambda info, x, c, stat: feat_freq_rank(
                    info, x['reference'], stat['target_vocab_rank'], side='target'
                ),
                require_training_set=True,
            ),
        }

        tok_features: dict[str, FeatureType] = {
            "tok_text": feature.Value(
                dtype="string",
                description="text of the token",
                func=lambda info, x, c: self._get_tok_text(c),
            ),
            "tok_capitalness": feature.Value(
                dtype="string",
                description="whether the token is capitalized",
                func=lambda info, x, c: cap_feature(c.features['tok_text']),
            ),
            "tok_position": feature.Value(
                dtype="float",
                description="relative position of token in sentence",
                func=self._get_tok_position,
            ),
            "tok_chars": feature.Value(
                dtype="float",
                description="number of characters in the token",
                func=lambda info, x, c: len(c.features['tok_text']),
            ),
            # TODO(gneubig): commented out because less important and harder to impl
            # "tok_test_freq": feature.Value(
            #     dtype="float",
            #     description="tok frequency in the test set",
            #     is_bucket=True,
            #     require_training_set=False,
            #     bucket_info=explainaboard.analysis.analyses.BucketAnalysis(
            #         method="bucket_attribute_specified_bucket_value",
            #         number=4,
            #         setting=(),
            #     ),
            # ),
            "tok_train_freq": feature.Value(
                dtype="float",
                description="tok frequency in the training set",
                require_training_set=True,
                func=lambda info, x, c, stat: stat['target_vocab'].get(
                    c.features['tok_text'], 0.0
                ),
            ),
        }

        return [
            AnalysisLevel(
                name='example',
                features=examp_features,
                metric_configs=self.default_metrics(level='example'),
            ),
            AnalysisLevel(
                name='token',
                features=tok_features,
                metric_configs=self.default_metrics(level='token'),
            ),
        ]

    def default_analyses(self) -> list[Analysis]:
        return self.continuous_feature_analyses()

    @classmethod
    def _get_default_eaas_strs(cls):
        return ['rouge1', 'rouge2', 'rougeL', 'bleu', 'length_ratio']

    @classmethod
    def default_metrics(
        cls, level='example', source_language=None, target_language=None
    ) -> list[MetricConfig]:
        eaas_defaults = cls._get_default_eaas_strs()
        metric_configs: list[Any] = []
        for metric_name in eaas_defaults:
            metric_configs.append(
                EaaSMetricConfig(
                    name=metric_name,
                    source_language=source_language,
                    target_language=target_language,
                )
            )

        defaults: dict[str, list] = {
            'example': metric_configs,
            'token': [
                F1ScoreConfig(
                    name='F1',
                    source_language=source_language,
                    target_language=target_language,
                )
            ],
        }
        return defaults[level]

    @classmethod
    def full_metric_list(
        cls, level='example', source_language=None, target_language=None
    ) -> list[MetricConfig]:
        full_metrics_eaas = [
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
        full_metrics_human = [
            "LikertScore_fluency",
            "LikertScore_coherence",
            "LikertScore_factuality",
        ]
        example_configs: list[MetricConfig] = []
        for x in full_metrics_eaas:
            example_configs.append(
                EaaSMetricConfig(
                    name=x,
                    source_language=source_language,
                    target_language=target_language,
                )
            )
        for x in full_metrics_human:
            example_configs.append(
                ExternalEvalConfig(
                    name=x,
                    aspect=x.split("LikertScore_")[1],
                )
            )
        defaults: dict[str, list[MetricConfig]] = {
            'example': example_configs,
            'tok': [
                F1ScoreConfig(
                    name='F1',
                    source_language=source_language,
                    target_language=target_language,
                )
            ],
        }
        return defaults[level]

    @staticmethod
    def _get_tok_position(info: SysOutputInfo, x: dict, c: AnalysisCase):
        ref_len = float(count_tokens(info, x['reference'], side='target'))
        if isinstance(c, AnalysisCaseMultiSpan):
            return cast(AnalysisCaseMultiSpan, c).spans[0].token_span[0] / ref_len
        elif isinstance(c, AnalysisCaseSpan):
            return cast(AnalysisCaseSpan, c).token_span[0] / ref_len
        else:
            raise ValueError(f'bad type {type(c)}')

    @staticmethod
    def _get_tok_text(c: AnalysisCase):
        if isinstance(c, AnalysisCaseMultiSpan):
            return cast(AnalysisCaseMultiSpan, c).spans[0].text
        elif isinstance(c, AnalysisCaseSpan):
            return cast(AnalysisCaseSpan, c).text
        else:
            raise ValueError(f'bad type {type(c)}')

    def _get_true_label(self, data_point: dict):
        return {'references': [data_point["reference"]], 'source': data_point["source"]}

    def _get_predicted_label(self, data_point: dict):
        return data_point["hypothesis"]

    def _gen_cases_and_stats(
        self,
        sys_info: SysOutputInfo,
        sys_output: list[dict],
        statistics: Any,
        analysis_level: AnalysisLevel,
    ) -> tuple[list[AnalysisCase], list[MetricStats]]:
        cases: list[AnalysisCase] = []
        if analysis_level.name == 'example':
            # Note that this is over-ridden to accommodate efficient calculation of
            # EaaS-style metrics

            inputs = []
            true_data = []
            pred_data = []
            for _id, feature_table in enumerate(sys_output):
                inputs.append(
                    {
                        "source": feature_table["source"],
                        "references": [feature_table["reference"]],
                        "hypothesis": feature_table["hypothesis"],
                    }
                )
                true_data.append(feature_table["reference"])
                pred_data.append(feature_table["hypothesis"])

            metric_names_eaas = []
            metric_configs_noneaas = []
            for metric_config in unwrap_generator(analysis_level.metric_configs):
                if isinstance(metric_config, EaaSMetricConfig):
                    metric_names_eaas.append(metric_config.name)
                else:
                    metric_configs_noneaas.append(metric_config)

            async_request = get_eaas_client().async_score(
                inputs,
                metrics=metric_names_eaas,
                calculate=['corpus', 'stats'],
            )

            metric_stats: list[Any] = [
                EaaSMetricStats(name=name, pos=i, eaas_request=async_request)
                for i, name in enumerate(metric_names_eaas)
            ]

            # For non-EaaS metrics
            for metric_config in metric_configs_noneaas:
                metric_stats.append(
                    metric_config.to_metric().calc_stats_from_data(
                        true_data, pred_data, metric_config
                    )
                )

            # Calculate features
            for i, output in progress(
                enumerate(sys_output), desc='calculating example-level features'
            ):
                case = AnalysisCase(sample_id=i, features={})
                for feat_name, feat_spec in analysis_level.features.items():
                    if feat_spec.func is None:
                        case.features[feat_name] = output[feat_name]
                    elif not feat_spec.require_training_set:
                        case.features[feat_name] = feat_spec.func(
                            sys_info, output, case
                        )
                    elif statistics is not None:
                        case.features[feat_name] = feat_spec.func(
                            sys_info, output, case, statistics
                        )
                cases.append(case)
        elif analysis_level.name == 'token':
            # Calculate features
            for i, output in progress(
                enumerate(sys_output), desc='calculating token-level features'
            ):
                # span features for true and predicted spans
                ref_toks = unwrap(sys_info.target_tokenizer)(output['reference'])
                hyp_toks = unwrap(sys_info.target_tokenizer)(output['hypothesis'])
                ref_feats = self._match_toks(ref_toks, hyp_toks)
                hyp_feats = self._match_toks(hyp_toks, ref_toks)
                # Get reference-only, hypothesis-only, and matched spans
                for ref_id, ref_info in enumerate(ref_feats):
                    ref_span = AnalysisCaseSpan(
                        sample_id=i,
                        token_span=ref_info['tok_pos'],
                        char_span=ref_info['tok_char_pos'],
                        orig_str='reference',
                        text=ref_info['tok_text'],
                        features={},
                    )
                    if ref_info['tok_matched'] < 0:
                        cases.append(ref_span)
                    else:
                        hyp_info = hyp_feats[ref_info['tok_matched']]
                        hyp_span = AnalysisCaseSpan(
                            sample_id=i,
                            token_span=hyp_info['tok_pos'],
                            char_span=hyp_info['tok_char_pos'],
                            orig_str='hypothesis',
                            text=hyp_info['tok_text'],
                            features={},
                        )
                        both_span = AnalysisCaseMultiSpan(
                            sample_id=i,
                            spans=[ref_span, hyp_span],
                            features={},
                        )
                        cases.append(both_span)
                for hyp_id, hyp_info in enumerate(hyp_feats):
                    if hyp_info['tok_matched'] < 0:
                        hyp_span = AnalysisCaseSpan(
                            sample_id=i,
                            token_span=hyp_info['tok_pos'],
                            char_span=hyp_info['tok_char_pos'],
                            orig_str='reference',
                            text=hyp_info['tok_text'],
                            features={},
                        )
                        cases.append(hyp_span)
            stats_list = []
            for case in cases:
                for feat_name, feat_spec in analysis_level.features.items():
                    if feat_spec.func is None:
                        raise ValueError(
                            f'could not find feature function for {feat_name}'
                        )
                    elif not feat_spec.require_training_set:
                        case.features[feat_name] = feat_spec.func(
                            sys_info, sys_output[case.sample_id], case
                        )
                    elif statistics is not None:
                        case.features[feat_name] = feat_spec.func(
                            sys_info, sys_output[case.sample_id], case, statistics
                        )
                # Both ref and hyp exist, so matched
                if isinstance(case, AnalysisCaseMultiSpan):
                    stats_list.append([1.0, 1.0, 1.0])
                elif cast(AnalysisCaseSpan, case).orig_str == 'reference':
                    stats_list.append([1.0, 0.0, 0.0])
                else:
                    stats_list.append([0.0, 1.0, 0.0])
            metric_stats = [SimpleMetricStats(np.array(stats_list))]
        else:
            raise ValueError(f'{analysis_level.name}-level analysis not supported')

        return cases, metric_stats

    @staticmethod
    def _match_toks(toks: TokenSeq, other_toks: TokenSeq):

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
            }
            # Save the features
            tok_dics.append(tok_dic)

        return tok_dics

    def _statistics_func(self, samples: Iterable[Any], sys_info: SysOutputInfo):
        samples_list = list(samples)
        source_vocab, source_vocab_rank = accumulate_vocab_from_samples(
            samples_list, lambda x: x['source'], unwrap(sys_info.source_tokenizer)
        )

        target_vocab, target_vocab_rank = accumulate_vocab_from_samples(
            samples_list, lambda x: x['reference'], unwrap(sys_info.target_tokenizer)
        )
        return {
            'source_vocab': source_vocab,
            'source_vocab_rank': source_vocab_rank,
            'target_vocab': target_vocab,
            'target_vocab_rank': target_vocab_rank,
        }
