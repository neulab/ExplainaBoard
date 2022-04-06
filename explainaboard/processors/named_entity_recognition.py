from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping
import re
from typing import Any, Optional

from datalabs import aggregating, Dataset
from tqdm import tqdm

from explainaboard import feature
from explainaboard.info import BucketPerformance, Performance, SysOutputInfo
import explainaboard.metric
from explainaboard.metric import Metric, MetricStats
from explainaboard.processors.processor import Processor
from explainaboard.processors.processor_registry import register_processor
from explainaboard.tasks import TaskType
from explainaboard.utils import bucketing, span_utils
from explainaboard.utils.analysis import cap_feature
from explainaboard.utils.py_utils import sort_dict
from explainaboard.utils.typing_utils import unwrap


@register_processor(TaskType.named_entity_recognition)
class NERProcessor(Processor):
    @classmethod
    def task_type(cls) -> TaskType:
        return TaskType.named_entity_recognition

    @classmethod
    def default_features(cls) -> feature.Features:
        return feature.Features(
            {
                "tokens": feature.Sequence(feature.Value("string")),
                "ner_true_tags": feature.Sequence(
                    feature.ClassLabel(
                        names=[
                            "O",
                            "B-PER",
                            "I-PER",
                            "B-ORG",
                            "I-ORG",
                            "B-LOC",
                            "I-LOC",
                            "B-MISC",
                            "I-MISC",
                        ]
                    )
                ),
                "ner_pred_tags": feature.Sequence(
                    feature.ClassLabel(
                        names=[
                            "O",
                            "B-PER",
                            "I-PER",
                            "B-ORG",
                            "I-ORG",
                            "B-LOC",
                            "I-LOC",
                            "B-MISC",
                            "I-MISC",
                        ]
                    )
                ),
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
                # --- the following are features of each entity ---
                "true_entity_info": feature.Sequence(
                    feature.Set(
                        {
                            "span_text": feature.Value("string"),
                            "span_len": feature.Value(
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
                            "span_position": feature.Value(
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
                            "span_density": feature.Value(
                                dtype="float",
                                description=(
                                    "Entity density. Given a sentence (or a sample), "
                                    "entity density tallies the ratio between the "
                                    "number of all entity tokens and tokens in this "
                                    "sentence"
                                ),
                                is_bucket=True,
                                bucket_info=feature.BucketInfo(
                                    method="bucket_attribute_specified_bucket_value",
                                    number=4,
                                    setting=(),
                                ),
                            ),
                            "econ": feature.Value(
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
                            "efre": feature.Value(
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

    @classmethod
    def default_metrics(cls) -> list[str]:
        return ["F1Score"]

    def _get_true_label(self, data_point: dict):
        return data_point["true_tags"]

    def _get_predicted_label(self, data_point: dict):
        return data_point["pred_tags"]

    def _get_statistics_resources(
        self, dataset_split: Dataset
    ) -> Optional[Mapping[str, Any]]:
        """
        From a DataLab dataset split, get resources necessary to calculate statistics
        """
        base_resources = unwrap(super()._get_statistics_resources(dataset_split))
        task_specific_resources = {
            'tag_id2str': dataset_split._info.task_templates[0].labels
        }
        return {**base_resources, **task_specific_resources}

    @aggregating()
    def _statistics_func(self, samples: Iterator, tag_id2str=None):
        """
        Input:
        samples: [{
         "tokens":
         "tags":
        }]
        Output:dict:
        """

        if tag_id2str is None:
            tag_id2str = []
        tokens_sequences = []
        tags_sequences = []
        tags_without_bio = list(
            set([t.split('-')[1].lower() if len(t) > 1 else t for t in tag_id2str])
        )

        vocab: dict[str, int] = {}
        for sample in tqdm(samples):
            tokens, tag_ids = sample["tokens"], sample["tags"]
            tags = [tag_id2str[tag_id] for tag_id in tag_ids]

            # update vocabulary
            for w in tokens:
                if w in vocab.keys():
                    vocab[w] += 1
                else:
                    vocab[w] = 1

            tokens_sequences += tokens
            tags_sequences += tags

        # efre_dic
        econ_dic = get_econ_dic(tokens_sequences, tags_sequences, tags_without_bio)
        # econ_dic = {"a":1} # for debugging purpose
        # econ_dic
        efre_dic = get_efre_dic(tokens_sequences, tags_sequences)
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

    # --- Feature functions accessible by ExplainaboardBuilder._get_feature_func()
    def _get_sentence_length(self, existing_features: dict):
        return len(existing_features["tokens"])

    def _get_stat_values(
        self, econ_dic: dict, efre_dic: dict, span_text: str, span_tag: str
    ):
        """
        Get entity consistency and frequency values
        """
        span_tag = span_tag.lower()
        span_text = span_text.lower()
        econ_val = 0.0
        if span_text in econ_dic and span_tag in econ_dic[span_text]:
            econ_val = float(econ_dic[span_text][span_tag])
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

    # --- End feature functions

    # These return none because NER is not yet in the main metric interface
    def _get_metrics(self, sys_info: SysOutputInfo) -> list[Metric]:
        return [
            getattr(explainaboard.metric, f'BIO{name}')()
            for name in unwrap(sys_info.metric_names)
        ]

    def _complete_span_features(self, sentence, tags, statistics=None):

        # Get training set stats if they exist
        has_stats = statistics is not None and len(statistics) > 0
        econ_dic = statistics["econ_dic"] if has_stats else None
        efre_dic = statistics["efre_dic"] if has_stats else None

        span_dics = []
        chunks = span_utils.get_spans_from_bio(tags)
        for tag, sid, eid in chunks:
            span_text = ' '.join(sentence[sid:eid])
            # Basic features
            span_dic = {
                'span_text': span_text,
                'span_len': eid - sid,
                'span_pos': (sid, eid),
                'span_tag': tag,
                'span_capitalness': cap_feature(span_text),
                'span_position': eid * 1.0 / len(sentence),
                'span_chars': len(span_text),
                'span_density': len(chunks) * 1.0 / len(sentence),
            }
            # Training set dependent features
            if has_stats:
                lower_tag = tag.lower()
                lower_text = span_text.lower()
                span_dic['econ'] = 0
                if span_text in econ_dic and lower_tag in econ_dic[lower_text]:
                    span_dic['econ'] = float(econ_dic[lower_text][lower_tag])
                span_dic['efre'] = efre_dic.get(lower_text, 0.0)
            # Save the features
            span_dics.append(span_dic)

        return span_dics

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
        for _id, dict_sysout in tqdm(enumerate(sys_output), desc="featurizing"):
            # Get values of bucketing features
            tokens = dict_sysout["tokens"]

            # sentence_length
            dict_sysout["sentence_length"] = len(tokens)

            # sentence-level training set dependent features
            if external_stats is not None:
                dict_sysout["num_oov"] = self._get_num_oov(tokens, external_stats)
                dict_sysout["fre_rank"] = self._get_fre_rank(tokens, external_stats)

            # span features for true and predicted spans
            dict_sysout["true_entity_info"] = self._complete_span_features(
                tokens, dict_sysout["true_tags"], statistics=external_stats
            )
            dict_sysout["pred_entity_info"] = self._complete_span_features(
                tokens, dict_sysout["pred_tags"], statistics=external_stats
            )
        # This is not used elsewhere, so just keep it as-is
        return list()

    def _get_span_ids(
        self,
        sys_output: list[dict],
        output_to_spans: Callable,
    ) -> Iterator[str]:
        for sample_id, my_output in enumerate(sys_output):
            for span_info in output_to_spans(my_output):
                span_text = span_info["span_text"]
                sid, eid = span_info["span_pos"]
                span_label = span_info["span_tag"]
                yield f'{sample_id}|||{sid}|||{eid}|||{span_text}|||{span_label}'

    def _get_span_sample_features(
        self,
        feature_name: str,
        sys_output: list[dict],
        output_to_spans: Callable,
    ) -> Iterator[str]:
        for sample_id, my_output in enumerate(sys_output):
            for _ in output_to_spans(my_output):
                yield my_output[feature_name]

    def _get_span_span_features(
        self,
        feature_name: str,
        sys_output: list[dict],
        output_to_spans: Callable,
    ) -> Iterator[str]:
        for sample_id, my_output in enumerate(sys_output):
            for span_info in output_to_spans(my_output):
                yield span_info[feature_name]

    def _bucketing_samples(
        self,
        sys_info: SysOutputInfo,
        sys_output: list[dict],
        active_features: list[str],
        metric_stats: list[MetricStats],
    ) -> tuple[dict, dict]:

        features = unwrap(sys_info.features)

        bucket_features = features.get_bucket_features()
        pcf_set = set(features.get_pre_computed_features())

        span_ids_true = list(
            self._get_span_ids(sys_output, lambda x: x["true_entity_info"])
        )
        span_ids_pred = list(
            self._get_span_ids(sys_output, lambda x: x["pred_entity_info"])
        )

        # Bucketing
        samples_over_bucket_true = {}
        samples_over_bucket_pred = {}
        performances_over_bucket = {}
        for feature_name in tqdm(bucket_features, desc="bucketing"):

            # Choose behavior based on whether this is a feature of samples or spans
            if feature_name in pcf_set:
                continue
            if feature_name in features:
                my_feature = features[feature_name]
                my_feature_func = self._get_span_sample_features
            else:
                my_feature = features["true_entity_info"].feature.feature[feature_name]
                my_feature_func = self._get_span_span_features
            bucket_info = my_feature.bucket_info

            # Get buckets for true spans
            bucket_func = getattr(bucketing, bucket_info.method)
            feat_vals = my_feature_func(
                feature_name, sys_output, lambda x: x["true_entity_info"]
            )
            feat_dict = {x: y for x, y in zip(span_ids_true, feat_vals)}
            samples_over_bucket_true[feature_name] = bucket_func(
                dict_obj=feat_dict,
                bucket_number=bucket_info.number,
                bucket_setting=bucket_info.setting,
            )

            # Get buckets for predicted spans
            feat_vals = my_feature_func(
                feature_name, sys_output, lambda x: x["pred_entity_info"]
            )
            feat_dict = {x: y for x, y in zip(span_ids_pred, feat_vals)}
            samples_over_bucket_pred[
                feature_name
            ] = bucketing.bucket_attribute_specified_bucket_interval(
                dict_obj=feat_dict,
                bucket_number=bucket_info.number,
                bucket_setting=samples_over_bucket_true[feature_name].keys(),
            )

            # evaluating bucket: get bucket performance
            performances_over_bucket[feature_name] = self.get_bucket_performance_ner(
                sys_info,
                sys_output,
                samples_over_bucket_true[feature_name],
                samples_over_bucket_pred[feature_name],
            )
        return samples_over_bucket_true, performances_over_bucket

    def _add_to_sample_dict(
        self,
        spans: list[str],
        type_id: str,
        sample_dict: defaultdict[str, dict[str, str]],
    ):
        """
        Get bucket samples (with mis-predicted entities) for each bucket given a feature
        (e.g., length)
        """
        for span in spans:
            split_span = span.split("|||")
            pos = "|||".join(split_span[0:4])
            tag = split_span[-1]
            sample_dict[pos][type_id] = tag

    def get_bucket_cases_ner(
        self,
        bucket_interval: str,
        sys_output: list[dict],
        samples_over_bucket_true: dict[str, list[str]],
        samples_over_bucket_pred: dict[str, list[str]],
    ) -> list:
        # Index samples for easy comparison
        sample_dict: defaultdict[str, dict[str, str]] = defaultdict(lambda: dict())
        self._add_to_sample_dict(
            samples_over_bucket_pred[bucket_interval], 'pred', sample_dict
        )
        self._add_to_sample_dict(
            samples_over_bucket_true[bucket_interval], 'true', sample_dict
        )

        case_list = []
        for pos, tags in sample_dict.items():
            true_label = tags.get('true', 'O')
            pred_label = tags.get('pred', 'O')

            split_pos = pos.split("|||")
            sent_id = int(split_pos[0])
            span = split_pos[-1]
            system_output_id = sys_output[int(sent_id)]["id"]
            error_case = {
                "span": span,
                "text": str(system_output_id),
                "true_label": true_label,
                "predicted_label": pred_label,
            }
            case_list.append(error_case)

        return case_list

    def get_bucket_performance_ner(
        self,
        sys_info: SysOutputInfo,
        sys_output: list[dict],
        samples_over_bucket_true: dict[str, list[str]],
        samples_over_bucket_pred: dict[str, list[str]],
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

        metric_names = unwrap(sys_info.metric_names)
        config = explainaboard.metric.F1ScoreConfig(ignore_classes=['O'])
        bucket_metrics = [
            getattr(explainaboard.metric, name)(config=config) for name in metric_names
        ]

        bucket_name_to_performance = {}
        for bucket_interval, spans_true in samples_over_bucket_true.items():

            if bucket_interval not in samples_over_bucket_pred.keys():
                raise ValueError("Predict Label Bucketing Errors")
            else:
                spans_pred = samples_over_bucket_pred[bucket_interval]

            """
            Get bucket samples for ner task
            """
            bucket_samples = self.get_bucket_cases_ner(
                bucket_interval,
                sys_output,
                samples_over_bucket_true,
                samples_over_bucket_pred,
            )

            true_labels = [x['true_label'] for x in bucket_samples]
            pred_labels = [x['predicted_label'] for x in bucket_samples]

            bucket_performance = BucketPerformance(
                bucket_name=bucket_interval,
                n_samples=len(spans_pred),
                bucket_samples=bucket_samples,
            )
            for metric in bucket_metrics:

                metric_val = metric.evaluate(
                    true_labels, pred_labels, conf_value=sys_info.conf_value
                )
                conf_low, conf_high = (
                    metric_val.conf_interval if metric_val.conf_interval else None,
                    None,
                )
                performance = Performance(
                    metric_name=metric.name,
                    value=metric_val.value,
                    confidence_score_low=conf_low,
                    confidence_score_high=conf_high,
                )
                bucket_performance.performances.append(performance)

            bucket_name_to_performance[bucket_interval] = bucket_performance

        return sort_dict(bucket_name_to_performance)


# TODO(gneubig): below is not done with refactoring
def get_econ_dic(train_word_sequences, tag_sequences_train, tags):
    """
    Note: when matching, the text span and tag have been lowercased.
    """
    econ_dic = dict()
    chunks_train = set(span_utils.get_spans_from_bio(tag_sequences_train))

    # print('tags: ', tags)
    count_idx = 0
    # print('num of computed chunks:', len(chunks_train))
    word_sequences_train_str = ' '.join(train_word_sequences).lower()
    for true_chunk in tqdm(chunks_train):
        # print('true_chunk',true_chunk)
        count_idx += 1
        # print()
        # print('progress: %d / %d: ' % (count_idx, len(chunks_train)))
        idx_start = true_chunk[1]
        idx_end = true_chunk[2]

        entity_span = ' '.join(train_word_sequences[idx_start:idx_end]).lower()
        if entity_span in econ_dic:
            continue
        else:
            econ_dic[entity_span] = dict()
            for tag in tags:
                econ_dic[entity_span][tag] = 0.0

        # Determine if the same position in pred list giving a right prediction.
        entity_span_new = ' ' + entity_span + ' '

        entity_span_new = (
            entity_span_new.replace('(', '')
            .replace(')', '')
            .replace('*', '')
            .replace('+', '')
        )
        # print('entity_span_new', entity_span_new)
        entity_str_sid = [
            m.start() for m in re.finditer(entity_span_new, word_sequences_train_str)
        ]
        # print('count_find_span:', len(entity_str_sid))
        if len(entity_str_sid) > 0:
            label_list = []
            # convert the string index into list index...
            entity_sids = []
            for str_idx in entity_str_sid:
                entity_sid = len(word_sequences_train_str[0:str_idx].split())
                entity_sids.append(entity_sid)
            entity_len = len(entity_span.split())

            for sid in entity_sids:
                label_candi_list = tag_sequences_train[sid : sid + entity_len]
                for label in label_candi_list:
                    klab = 'o'
                    if len(label.split('-')) > 1:
                        klab = label.split('-')[1].lower()
                    label_list.append(klab)

            label_norep = list(set(label_list))
            for lab_norep in label_norep:
                hard = float(
                    '%.3f' % (float(label_list.count(lab_norep)) / len(label_list))
                )
                econ_dic[entity_span][lab_norep] = hard

    # fwrite = open(path_write, 'wb')
    # pickle.dump(econ_dic, fwrite)
    # fwrite.close()
    """
    {
        'benson koech': {'O': 0.0, 'org': 0.0, 'loc': 0.0, 'per': 1.0, 'misc': 0.0}
    }
    """
    # exit()
    return econ_dic


# Global functions for training set dependent features
def get_efre_dic(train_word_sequences, tag_sequences_train):
    efre_dic = dict()
    chunks_train = set(span_utils.get_spans_from_bio(tag_sequences_train))
    count_idx = 0
    word_sequences_train_str = ' '.join(train_word_sequences).lower()
    for true_chunk in tqdm(chunks_train):
        count_idx += 1
        # print('progress: %d / %d: ' % (count_idx, len(chunks_train)))
        idx_start = true_chunk[1]
        idx_end = true_chunk[2]

        entity_span = ' '.join(train_word_sequences[idx_start:idx_end]).lower()
        # print('entity_span', entity_span)
        if entity_span in efre_dic:
            continue
        else:
            efre_dic[entity_span] = []

        # Determine if the same position in pred list giving a right prediction.
        entity_span_new = ' ' + entity_span + ' '
        entity_span_new = (
            entity_span_new.replace('(', '')
            .replace(')', '')
            .replace('*', '')
            .replace('+', '')
        )
        entity_str_sid = [
            m.start() for m in re.finditer(entity_span_new, word_sequences_train_str)
        ]

        efre_dic[entity_span] = len(entity_str_sid)

    sorted_efre_dic = sorted(efre_dic.items(), key=lambda item: item[1], reverse=True)

    efre_dic_keep = {}
    count_bigger_than_max_freq = 0
    max_freq = float(sorted_efre_dic[4][1])
    for span, freq in efre_dic.items():
        if freq <= max_freq:
            efre_dic_keep[span] = '%.3f' % (float(freq) / max_freq)
        else:
            count_bigger_than_max_freq += 1

    return efre_dic_keep
