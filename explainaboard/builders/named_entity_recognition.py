import re
from typing import Callable, Tuple
from typing import Iterator, Dict, List

from datalabs import load_dataset
from datalabs.operations.aggregate.sequence_labeling import (
    sequence_labeling_aggregating,
)
from explainaboard.utils.eval_bucket import f1_score_seqeval_bucket
from tqdm import tqdm

import explainaboard.utils.bucketing
from explainaboard.builders import ExplainaboardBuilder
from explainaboard.info import SysOutputInfo, BucketPerformance, Performance
from explainaboard.utils.analysis import *
from explainaboard.utils.py_utils import eprint
from explainaboard.utils.eval_basic import get_chunks, f1_score_seqeval


def get_econ_dic(train_word_sequences, tag_sequences_train, tags):
    """
    Note: when matching, the text span and tag have been lowercased.
    """
    econ_dic = dict()
    chunks_train = set(get_chunks(tag_sequences_train))

    print('tags: ', tags)
    count_idx = 0
    print('num of computed chunks:', len(chunks_train))
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
                label_candi_list = tag_sequences_train[sid:sid + entity_len]
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


def get_efre_dic(train_word_sequences, tag_sequences_train):
    efre_dic = dict()
    chunks_train = set(get_chunks(tag_sequences_train))
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
    # print('The number of words whose word frequency exceeds the threshold: %d is %d' % (
    #     max_freq, count_bigger_than_max_freq))

    # fwrite = open(path_write, 'wb')
    # pickle.dump(efre_dic_keep, fwrite)
    # fwrite.close()

    return efre_dic_keep


@sequence_labeling_aggregating(
    name="get_statistics",
    contributor="datalab",
    task="sequence-labeling, named-entity-recognition, structure-prediction",
    description="Calculate the overall statistics (e.g., average length) of "
    "a given sequence labeling datasets (e.g., named entity recognition)",
)
def get_statistics(samples: Iterator, tag_id2str=None):
    """
    Input:
    samples: [{
     "tokens":
     "tags":
    }]
    Output:dict:
    """

    # tag_id2str = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

    if tag_id2str is None:
        tag_id2str = []
    tokens_sequences = []
    tags_sequences = []
    tags_without_bio = list(
        set([t.split('-')[1].lower() if len(t) > 1 else t for t in tag_id2str])
    )

    vocab = {}
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


class NERExplainaboardBuilder(ExplainaboardBuilder):
    def __init__(self):
        super().__init__()
        self._statistics_func = get_statistics

    def _init_statistics(self, sys_info: SysOutputInfo, statistics_func: Callable):
        """Take in information about the system outputs and a statistic calculating function and return a dictionary
        of statistics.

        :param sys_info: Information about the system outputs
        :param statistics_func: The function used to get the statistics
        :return: Statistics from, usually, the training set that are used to calculate other features
        """

        # TODO(gneubig): this is a bit different than others, and probably should override the parent class
        # Calculate statistics of training set
        eprint(sys_info.dataset_name, sys_info.sub_dataset_name)
        statistics = None
        if sys_info.dataset_name is not None:
            try:

                dataset = load_dataset(sys_info.dataset_name, sys_info.sub_dataset_name)
                if (
                    len(dataset['train']._stat) == 0 or not sys_info.reload_stat
                ):  # calculate the statistics (_stat) when _stat is {} or `reload_stat` is False
                    tag_id2str = dataset['train']._info.task_templates[0].labels
                    statistics_func.resources = {"tag_id2str": tag_id2str}
                    new_train = dataset['train'].apply(statistics_func, mode="local")
                    statistics = new_train._stat
                else:
                    statistics = dataset["train"]._stat
            except FileNotFoundError:
                eprint(
                    "The dataset hasn't been supported by DataLab so no training set dependent features will be supported by ExplainaBoard." # noqa
                    "You can add the dataset by: https://github.com/ExpressAI/DataLab/blob/main/docs/SDK/add_new_datasets_into_sdk.md" # noqa
                )
        return statistics

    # --- Feature functions accessible by ExplainaboardBuilder._get_feature_func()
    def _get_sentence_length(self, existing_features: dict):
        return len(existing_features["tokens"])

    def _get_econ_value(self, span_dic: dict, span_text: str, span_tag: str):
        """
        Since keys and values of span_dic have been lower-cased, we also need to lowercase span_tag and span_text

        """
        span_tag = span_tag.lower()
        span_text = span_text.lower()

        econ_value = 0.0
        if span_text in span_dic.keys():
            if span_tag in span_dic[span_text]:
                econ_value = float(span_dic[span_text][span_tag])
        return econ_value

    def _get_efre_value(self, span_dic, span_text):
        efre_value = 0.0
        span_text = span_text.lower()
        if span_text in span_dic.keys():
            efre_value = float(span_dic[span_text])
        return efre_value

    # training set dependent features
    def _get_num_oov(self, tokens, statistics):
        num_oov = 0

        for w in tokens:
            if w not in statistics['vocab'].keys():
                num_oov += 1
        # print(num_oov)
        return num_oov

    # training set dependent features (this could be merged into the above one for further optimization)
    def _get_fre_rank(self, tokens, statistics):
        fre_rank = 0

        for w in tokens:
            if w not in statistics['vocab_rank'].keys():
                fre_rank += len(statistics['vocab_rank'])
            else:
                fre_rank += statistics['vocab_rank'][w]

        fre_rank = 0 if len(tokens) == 0 else fre_rank * 1.0 / len(tokens)
        return fre_rank

    # --- End feature functions

    def _complete_feature_raw_span_features(self, sentence, tags):
        # span_text, span_len, span_pos, span_tag
        chunks = get_chunks(tags)
        span_dics = []
        for chunk in chunks:
            tag, sid, eid = chunk
            # span_text = ' '.join(sentence[sid:eid]).lower()
            span_text = ' '.join(sentence[sid:eid])
            span_len = eid - sid
            span_pos = (sid, eid)
            span_dic = {
                'span_text': span_text,
                'span_len': span_len,
                'span_pos': span_pos,
                'span_tag': tag,
                'span_capitalness': cap_feature(span_text),  # noqa
                'span_position': eid * 1.0 / len(sentence),
                'span_chars': len(span_text),
                'span_density': len(chunks) * 1.0 / len(sentence),
            }
            # print('span_dic: ',span_dic)
            span_dics.append(span_dic)
        # self.span_dics = span_dics
        return span_dics

    def _complete_feature_advanced_span_features(self, sentence, tags, statistics=None):
        span_dics = self._complete_feature_raw_span_features(sentence, tags)
        # if not self.dict_pre_computed_models:
        #     return span_dics

        if (
            statistics is None or len(statistics) == 0
        ):  # there is no training set dependent features
            return span_dics

        # econ_dic = self.dict_pre_computed_models['econ']
        # econ_dic = statistics["econ_dic"]
        econ_dic = statistics["econ_dic"]
        # efre_dic = self.dict_pre_computed_models['efre']
        efre_dic = statistics["efre_dic"]

        for span_dic in span_dics:
            span_text = span_dic['span_text']
            span_tag = span_dic['span_tag']

            # compute the entity-level label consistency

            if 'econ' not in span_dic:
                span_dic['econ'] = self._get_econ_value(econ_dic, span_text, span_tag)
            # compute the entity-level frequency
            if 'efre' not in span_dic:
                span_dic['efre'] = self._get_efre_value(efre_dic, span_text)

        return span_dics

    # TODO(gneubig): can this be generalized or is it specialized?
    def _complete_features(
        self, sys_info: SysOutputInfo, sys_output: List[dict], statistics=None
    ) -> List[str]:
        """
        This function takes in meta-data about system outputs, system outputs, and a few other optional pieces of
        information, then calculates feature functions and modifies `sys_output` to add these feature values

        :param sys_info: Information about the system output
        :param sys_output: The system output itself
        :param statistics: Training set statistics that are used to calculate training set specific features
        :return: The features that are active (e.g. skipping training set features when no training set available)
        """
        for _id, dict_sysout in tqdm(enumerate(sys_output), desc="featurizing"):
            # Get values of bucketing features
            tokens = dict_sysout["tokens"]

            # sentence_length
            dict_sysout["sentence_length"] = len(tokens)

            # sentence-level training set dependent features
            if statistics is not None:
                dict_sysout["num_oov"] = self._get_num_oov(tokens, statistics)
                dict_sysout["fre_rank"] = self._get_fre_rank(tokens, statistics)

            dict_sysout[
                "true_entity_info"
            ] = self._complete_feature_advanced_span_features(
                tokens, dict_sysout["true_tags"], statistics=statistics
            )
            dict_sysout[
                "pred_entity_info"
            ] = self._complete_feature_advanced_span_features(
                tokens, dict_sysout["pred_tags"], statistics=statistics
            )
        # This should return a list, but this list isn't used in the overridden function so ignore for now
        return None # noqa

    # TODO(gneubig): should this be generalized or is it task specific?
    def get_overall_performance(
        self,
        sys_info: SysOutputInfo,
        sys_output: List[dict],
    ) -> Dict[str, Performance]:
        predicted_labels, true_labels = [], []  # noqa

        true_tags_list = []
        pred_tags_list = []

        for _id, feature_table in enumerate(sys_output):

            true_tags_list.append(feature_table["true_tags"])
            pred_tags_list.append(feature_table["pred_tags"])

        overall = {}
        for metric_name in sys_info.metric_names:
            if metric_name != 'f1_score_seqeval':
                raise NotImplementedError(f'Unsupported metric {metric_name}')
            res_json = f1_score_seqeval(true_tags_list, pred_tags_list)

            overall_value = res_json["f1"]
            # overall_value = f1_score_seqeval(true_tags_list, pred_tags_list)["f1"]

            # metric_name = "F1score_seqeval"
            overall_performance = Performance(
                metric_name=metric_name,
                value=overall_value,
                confidence_score_low=0.0,
                confidence_score_up=0.0,
            )
            overall[metric_name] = overall_performance
        return overall

    def _bucketing_samples_add_feats(
        self,
        sample_id,
        bucket_features,
        pcf_set,
        feature_table,
        entity_info_list,
        feature_to_sample_address_to_value,
    ):

        for span_info in entity_info_list:
            span_text = span_info["span_text"]
            span_pos = span_info["span_pos"]
            span_label = span_info["span_tag"]

            span_address = (
                f'{sample_id}|||{span_pos[0]}|||{span_pos[1]}|||{span_text}|||{span_label}'
            )

            for feature_name in bucket_features:
                if feature_name not in feature_to_sample_address_to_value.keys():
                    feature_to_sample_address_to_value[feature_name] = {}
                elif feature_name in feature_table:  # first-level features
                    feature_to_sample_address_to_value[feature_name][
                        span_address
                    ] = feature_table[feature_name]
                elif feature_name in span_info:  # second-level features
                    feature_to_sample_address_to_value[feature_name][
                        span_address
                    ] = span_info[feature_name]
                elif feature_name not in pcf_set:
                    raise ValueError(
                        f'Missing feature {feature_name} not found and not pre-computed'
                    )

    def _bucketing_samples(
        self,
        sys_info: SysOutputInfo,
        sys_output: List[dict],
        active_features: List[str],
    ) -> Tuple[dict, dict]:

        sample_address = ""  # noqa
        feature_to_sample_address_to_value_true = {}
        feature_to_sample_address_to_value_pred = {}

        bucket_features = sys_info.features.get_bucket_features()
        pcf_set = set(sys_info.features.get_pre_computed_features())

        # Preparation for bucketing
        for _id, feature_table in enumerate(sys_output):

            self._bucketing_samples_add_feats(
                _id,
                bucket_features,
                pcf_set,
                feature_table,
                feature_table["true_entity_info"],
                feature_to_sample_address_to_value_true,
            )
            self._bucketing_samples_add_feats(
                _id,
                bucket_features,
                pcf_set,
                feature_table,
                feature_table["pred_entity_info"],
                feature_to_sample_address_to_value_pred,
            )

        # Bucketing
        samples_over_bucket = {}
        samples_over_bucket_pred = {}
        performances_over_bucket = {}
        for feature_name in tqdm(
            sys_info.features.get_bucket_features(), desc="bucketing"
        ):

            _bucket_info = ""
            if feature_name in sys_info.features.keys():
                _bucket_info = sys_info.features[feature_name].bucket_info
            else:
                # print(sys_info.features)
                _bucket_info = (
                    sys_info.features["true_entity_info"]
                    .feature.feature[feature_name]
                    .bucket_info
                )

            # The following indicates that there are no examples, probably because necessary data for bucketing
            # was not available.
            if (
                len(feature_to_sample_address_to_value_true[feature_name]) == 0
                or len(feature_to_sample_address_to_value_pred[feature_name]) == 0
            ):
                continue

            bucket_func = getattr(
                explainaboard.utils.bucketing,
                _bucket_info.method,
            )
            samples_over_bucket[feature_name] = bucket_func(
                dict_obj=feature_to_sample_address_to_value_true[feature_name],
                bucket_number=_bucket_info.number,
                bucket_setting=_bucket_info.setting,
            )

            # print(f"debug-1: {samples_over_bucket_true[feature_name]}")
            samples_over_bucket_pred[
                feature_name
            ] = explainaboard.utils.bucketing.bucket_attribute_specified_bucket_interval(
                dict_obj=feature_to_sample_address_to_value_pred[feature_name],
                bucket_number=_bucket_info.number,
                bucket_setting=samples_over_bucket[feature_name].keys(),
            )

            # print(f"samples_over_bucket.keys():\n{samples_over_bucket_true.keys()}")

            # evaluating bucket: get bucket performance
            performances_over_bucket[feature_name] = self.get_bucket_performance_ner(
                sys_info,
                sys_output,
                samples_over_bucket[feature_name],
                samples_over_bucket_pred[feature_name],
            )
        return samples_over_bucket, performances_over_bucket

    """
    Get bucket samples (with mis-predicted entities) for each bucket given a feature (e.g., length)
    """

    def _create_sample_dict(self, samp_bucket, interval):
        dict_pos2tag = {}
        for k_bucket_eval, spans in samp_bucket.items():
            if k_bucket_eval != interval:
                continue
            for span in spans:
                pos = "|||".join(span.split("|||")[0:4])
                tag = span.split("|||")[-1]
                dict_pos2tag[pos] = tag
        return dict_pos2tag

    def get_bucket_cases_ner(
        self,
        bucket_interval,
        sys_output: List[dict],
        samples_over_bucket: Dict[str, List[str]],
        samples_over_bucket_pred: Dict[str, List[str]],
    ) -> list:
        # predict:  2_3 -> NER
        dict_pos2tag_pred = self._create_sample_dict(samples_over_bucket_pred, bucket_interval)
        dict_pos2tag = self._create_sample_dict(samples_over_bucket, bucket_interval)

        error_case_list = []
        for pos, tag in dict_pos2tag.items():

            true_label = tag
            sent_id = int(pos.split("|||")[0])
            span = pos.split("|||")[-1]
            system_output_id = sys_output[int(sent_id)]["id"]

            if pos in dict_pos2tag_pred.keys():
                pred_label = dict_pos2tag_pred[pos]
                if true_label == pred_label:
                    continue
            else:
                pred_label = "O"
            error_case = {
                "span": span,
                "text": str(system_output_id),
                "true_label": true_label,
                "predicted_label": pred_label,
            }
            error_case_list.append(error_case)

        for pos, tag in dict_pos2tag_pred.items():

            pred_label = tag

            sent_id = int(pos.split("|||")[0])
            span = pos.split("|||")[-1]
            span_sentence = " ".join(sys_output[sent_id]["tokens"])  # noqa
            system_output_id = sys_output[int(sent_id)]["id"]
            # print(span_sentence)

            if pos in dict_pos2tag.keys():
                true_label = dict_pos2tag[pos]
                if true_label == pred_label:
                    continue
            else:
                true_label = "O"
            # error_case = span + "|||" + span_sentence + "|||" + true_label + "|||" + pred_label
            error_case = {
                "span": span,
                "text": system_output_id,
                "true_label": true_label,
                "predicted_label": pred_label,
            }
            error_case_list.append(error_case)

        return error_case_list

    def get_bucket_performance_ner(
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

            for metric_name in sys_info.metric_names:
                """
                # Note that: for NER task, the bucket-wise evaluation function is a little different from overall
                #            evaluation function
                # for overall: f1_score_seqeval
                # for bucket:  f1_score_seqeval_bucket
                """
                if metric_name != 'f1_score_seqeval':
                    raise NotImplementedError(f'Unsupported metric {metric_name}')
                f1, p, r = f1_score_seqeval_bucket(spans_pred, spans_true)

                bucket_name_to_performance[bucket_interval] = []
                bucket_performance = BucketPerformance(
                    bucket_name=bucket_interval,
                    metric_name=metric_name,
                    value=f1,
                    confidence_score_low=0.0,
                    confidence_score_up=0.0,
                    n_samples=len(spans_pred),
                    bucket_samples=bucket_samples,
                )

                bucket_name_to_performance[bucket_interval].append(bucket_performance)

        return sort_dict(bucket_name_to_performance)  # noqa
