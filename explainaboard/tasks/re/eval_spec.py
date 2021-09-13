# -*- coding: utf-8 -*-
import explainaboard.error_analysis as ea
import numpy
import os


def get_aspect_value(sample_list, dict_aspect_func):
    dict_span2aspect_val = {}
    dict_span2aspect_val_pred = {}

    for aspect, fun in dict_aspect_func.items():
        dict_span2aspect_val[aspect] = {}
        dict_span2aspect_val_pred[aspect] = {}

    # maintain it for print error case
    dict_sid2sent = {}

    sample_id = 0
    for info_list in sample_list:

        #
        #
        #
        # word_list = word_segment(sent).split(" ")

        # Sentence	Entities	Paragraph	True Relation Label	Predicted Relation Label
        # Sentence Length	Paragraph Length	Number of Entities in Ground Truth Relation	Average Distance of Entities

        sent, entities, paragraph, true_label, pred_label, sent_length, para_length, n_entity, avg_distance = info_list

        dict_sid2sent[str(sample_id)] = ea.format4json2(entities + "|||" + sent)

        sent_pos = ea.tuple2str((sample_id, true_label))
        sent_pos_pred = ea.tuple2str((sample_id, pred_label))

        # Sentence Length: sentALen
        aspect = "sLen"
        if aspect in dict_aspect_func.keys():
            dict_span2aspect_val[aspect][sent_pos] = float(sent_length)
            dict_span2aspect_val_pred[aspect][sent_pos_pred] = float(sent_length)

        # Paragraph Length: pLen
        aspect = "pLen"
        if aspect in dict_aspect_func.keys():
            dict_span2aspect_val[aspect][sent_pos] = float(para_length)
            dict_span2aspect_val_pred[aspect][sent_pos_pred] = float(para_length)

        # Number of Entity: nEnt
        aspect = "nEnt"
        if aspect in dict_aspect_func.keys():
            dict_span2aspect_val[aspect][sent_pos] = float(n_entity)
            dict_span2aspect_val_pred[aspect][sent_pos_pred] = float(n_entity)

        # Average Distance: avgDist
        aspect = "avgDist"
        if aspect in dict_aspect_func.keys():
            dict_span2aspect_val[aspect][sent_pos] = float(avg_distance)
            dict_span2aspect_val_pred[aspect][sent_pos_pred] = float(avg_distance)

        # Tag: tag
        aspect = "tag"  ############## MUST Be Gold Tag for text classification task
        if aspect in dict_aspect_func.keys():
            dict_span2aspect_val[aspect][sent_pos] = true_label
            dict_span2aspect_val_pred[aspect][sent_pos_pred] = true_label

        sample_id += 1

    # print(dict_span2aspect_val["bleu"])
    return dict_span2aspect_val, dict_span2aspect_val_pred, dict_sid2sent


def evaluate(task_type="ner", analysis_type="single", systems=[], output_filename="./output.json", is_print_ci=False,
             is_print_case=False, is_print_ece=False):

    path_text = systems[0] if analysis_type == "single" else ""
    path_comb_output = "model_name" + "/" + path_text.split("/")[-1]
    dict_aspect_func, dict_precomputed_path, obj_json = ea.load_task_conf(task_dir=os.path.dirname(__file__))

    sample_list, sent_list, entity_list, true_list, pred_list = file_to_list(path_text)

    error_case_list = []
    if is_print_case:
        error_case_list = get_error_case(sent_list, entity_list, true_list, pred_list)
        print(" -*-*-*- the number of error casse:\t", len(error_case_list))

    dict_span2aspect_val, dict_span2aspect_val_pred, dict_sid2sent = get_aspect_value(sample_list, dict_aspect_func)

    holistic_performance = ea.accuracy(true_list, pred_list)
    holistic_performance = format(holistic_performance, '.3g')

    # Confidence Interval of Holistic Performance
    confidence_low, confidence_up = 0, 0
    if is_print_ci:
        confidence_low, confidence_up = ea.compute_confidence_interval_acc(true_list, pred_list, n_times=1000)

    dict_span2aspect_val, dict_span2aspect_val_pred, dict_sid2sent = get_aspect_value(sample_list, dict_aspect_func)

    print("------------------ Holistic Result----------------------")
    print(holistic_performance)

    # print(f1(list_true_tags_token, list_pred_tags_token)["f1"])

    dict_bucket2span = {}
    dict_bucket2span_pred = {}
    dict_bucket2f1 = {}
    aspect_names = []

    for aspect, func in dict_aspect_func.items():
        # print(aspect, dict_span2aspect_val[aspect])
        dict_bucket2span[aspect] = ea.select_bucketing_func(func[0], func[1], dict_span2aspect_val[aspect])
        # print(aspect, dict_bucket2span[aspect])
        # exit()
        dict_bucket2span_pred[aspect] = ea.bucket_attribute_specified_bucket_interval(dict_span2aspect_val_pred[aspect],
                                                                                      dict_bucket2span[aspect].keys())
        # dict_bucket2span_pred[aspect] = __select_bucketing_func(func[0], func[1], dict_span2aspect_val_pred[aspect])
        dict_bucket2f1[aspect] = get_bucket_acc_with_error_case(dict_bucket2span[aspect],
                                                                dict_bucket2span_pred[aspect], dict_sid2sent,
                                                                is_print_ci, is_print_case)
        aspect_names.append(aspect)
    print("aspect_names: ", aspect_names)

    print("------------------ Breakdown Performance")
    for aspect in dict_aspect_func.keys():
        ea.print_dict(dict_bucket2f1[aspect], aspect)
    print("")

    # Calculate databias w.r.t numeric attributes
    dict_aspect2bias = {}
    for aspect, aspect2Val in dict_span2aspect_val.items():
        if type(list(aspect2Val.values())[0]) != type("string"):
            dict_aspect2bias[aspect] = numpy.average(list(aspect2Val.values()))

    print("------------------ Dataset Bias")
    for k, v in dict_aspect2bias.items():
        print(k + ":\t" + str(v))
    print("")

    def beautify_interval(interval):

        if type(interval[0]) == type("string"):  ### pay attention to it
            return interval[0]
        else:
            if len(interval) == 1:
                bk_name = '(' + format(float(interval[0]), '.3g') + ',)'
                return bk_name
            else:
                range1_r = '(' + format(float(interval[0]), '.3g') + ','
                range1_l = format(float(interval[1]), '.3g') + ')'
                bk_name = range1_r + range1_l
                return bk_name

    dict_fine_grained = {}
    for aspect, metadata in dict_bucket2f1.items():
        dict_fine_grained[aspect] = []
        for bucket_name, v in metadata.items():
            # print("---------debug--bucket name old---")
            # print(bucket_name)
            bucket_name = beautify_interval(bucket_name)
            # print("---------debug--bucket name new---")
            # print(bucket_name)

            # bucket_value = format(v[0]*100,'.4g')
            bucket_value = format(v[0], '.4g')
            n_sample = v[1]
            confidence_low_bucket = format(v[2], '.4g')
            confidence_up_bucket = format(v[3], '.4g')
            bucket_error_case = v[4]

            # instantiation
            dict_fine_grained[aspect].append({"bucket_name": bucket_name, "bucket_value": bucket_value, "num": n_sample,
                                             "confidence_low": confidence_low_bucket,
                                             "confidence_up": confidence_up_bucket,
                                             "bucket_error_case": bucket_error_case})

    obj_json["task"] = task_type
    obj_json["data"]["language"] = "English"
    obj_json["data"]["bias"] = dict_aspect2bias
    obj_json["data"]["output"] = path_comb_output

    obj_json["model"]["results"]["overall"]["error_case"] = error_case_list
    obj_json["model"]["results"]["overall"]["performance"] = holistic_performance
    obj_json["model"]["results"]["overall"]["confidence_low"] = confidence_low
    obj_json["model"]["results"]["overall"]["confidence_up"] = confidence_up
    obj_json["model"]["results"]["fine_grained"] = dict_fine_grained

    raise NotImplementedError('RE is not fully implemented yet, see below')

    # ece = 0
    # dic_calibration = None
    # if is_print_ece:
    #     ece, dic_calibration = process_all(path_text,
    #                                        size_of_bin=10, dataset=corpus_type, model=model_name)

    # obj_json["model"]["results"]["calibration"] = dic_calibration
    # # print(dic_calibration)

    # ea.save_json(obj_json, output_filename)

#
# def main():
#
# 	parser = argparse.ArgumentParser(description='Interpretable Evaluation for NLP')
#
#
# 	parser.add_argument('--task', type=str, required=True,
# 						help="absa")
#
# 	parser.add_argument('--ci', type=str, required=False, default= False,
# 						help="True|False")
#
# 	parser.add_argument('--case', type=str, required=False, default= False,
# 						help="True|False")
#
# 	parser.add_argument('--ece', type=str, required=False, default= False,
# 						help="True|False")
#
#
# 	parser.add_argument('--type', type=str, required=False, default="single",
# 						help="analysis type: single|pair|combine")
# 	parser.add_argument('--systems', type=str, required=True,
# 						help="the directories of system outputs. Multiple one should be separated by comma, for example, system1,system2 (no space)")
#
# 	parser.add_argument('--output', type=str, required=True,
# 						help="analysis output file")
# 	args = parser.parse_args()
#
#
# 	is_print_ci = args.ci
# 	is_print_case = args.case
# 	is_print_ece = args.ece
#
# 	task = args.task
# 	analysis_type = args.type
# 	systems = args.systems.split(",")
# 	output = args.output
#
#
# 	print("task", task)
# 	print("type", analysis_type)
# 	print("systems", systems)
# 	# sample_list = file_to_list_re(systems[0])
# 	# print(sample_list[0])
# 	evaluate(task_type=task, analysis_type=analysis_type, systems=systems, output=output, is_print_ci = is_print_ci, is_print_case = is_print_case, is_print_ece = is_print_ece)
#
# # python eval_spec.py  --task re --systems ./test_re.tsv --output ./a.json
# if __name__ == '__main__':
# 	main()
def get_bucket_acc_with_error_case(dict_bucket2span, dict_bucket2span_pred, dict_sid2sent, is_print_ci, is_print_case):
    # The structure of span_true or span_pred
    # 2345|||Positive
    # 2345 represents sentence id
    # Positive represents the "label" of this instance

    dict_bucket2f1 = {}

    for bucket_interval, spans_true in dict_bucket2span.items():
        spans_pred = []
        if bucket_interval not in dict_bucket2span_pred.keys():
            raise ValueError("Predict Label Bucketing Errors")
        else:
            spans_pred = dict_bucket2span_pred[bucket_interval]

        # loop over samples from a given bucket
        error_case_bucket_list = []
        if is_print_case:
            for info_true, info_pred in zip(spans_true, spans_pred):
                sid_true, label_true = info_true.split("|||")
                sid_pred, label_pred = info_pred.split("|||")
                if sid_true != sid_pred:
                    continue

                sent_entities = dict_sid2sent[sid_true]
                if label_true != label_pred:
                    error_case_info = label_true + "|||" + label_pred + "|||" + sent_entities
                    error_case_bucket_list.append(error_case_info)

        accuracy_each_bucket = ea.accuracy(spans_pred, spans_true)
        confidence_low, confidence_up = 0, 0
        if is_print_ci:
            confidence_low, confidence_up = ea.compute_confidence_interval_acc(spans_pred, spans_true)
        dict_bucket2f1[bucket_interval] = [accuracy_each_bucket, len(spans_true), confidence_low, confidence_up,
                                           error_case_bucket_list]

    return ea.sort_dict(dict_bucket2f1)


def get_error_case(sent_list, entity_list, true_label_list, pred_label_list):
    error_case_list = []
    for sent, entities, true_label, pred_label in zip(sent_list, entity_list, true_label_list, pred_label_list):
        if true_label != pred_label:
            error_case_list.append(true_label + "|||" + pred_label + "|||" + entities + "|||" + ea.format4json2(sent))
    return error_case_list


def file_to_list(file_path):
    sample_list = []
    fin = open(file_path, "r")
    true_list = []
    pred_list = []
    sent_list = []
    entity_list = []
    for idx, line in enumerate(fin):
        if idx == 0:
            continue
        info_list = line.rstrip("\n").split("\t")
        sample_list.append([info for info in info_list])
        true_list.append(info_list[3])
        pred_list.append(info_list[4])
        sent_list.append(info_list[0])
        entity_list.append(info_list[1])

    return sample_list, sent_list, entity_list, true_list, pred_list