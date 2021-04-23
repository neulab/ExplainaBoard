import csv


def get_probability_right_or_not(file_path):
    """

    :param file_path: the file_path is the path to your file.

    And the path must include file name.

    the file name is in this format: test_dataset_model.tsv.

    the file_path must in the format: /root/path/to/your/file/test_dataset.tsv

    The file must in this format:
    sentence\tground_truth\tpredict_label\tprobability\tright_or_not
    if prediction is right, right_or_not is assigned to 1, otherwise 0.

    """

    import pandas as pd
    import numpy as np

    result = pd.read_csv(file_path, sep='\t', header=None)

    probability_list = np.array(result[3]).tolist()
    right_or_not_list = np.array(result[4]).tolist()

    return probability_list, right_or_not_list


def get_raw_list(probability_list, right_or_not_list):
    total_raw_list = []

    for index in range(len(right_or_not_list)):
        total_raw_list.append([probability_list[index], right_or_not_list[index]])
    return total_raw_list


def calculate_ece(result_list):
    ece = 0
    size = 0
    tem_list = []
    for value in result_list:
        if value[2] == 0:
            tem_list.append(0)
            continue
        size = size + value[2]
        error = abs(float(value[0]) - float(value[1]))
        tem_list.append(error)

    if size == 0:
        return -1

    for i in range(len(result_list)):
        ece = ece + ((result_list[i][2] / size) * tem_list[i])

    return ece


def divide_into_bin(size_of_bin, raw_list):
    bin_list = []
    basic_width = 1 / size_of_bin

    for i in range(0, size_of_bin):
        bin_list.append([])

    for value in raw_list:
        probability = value[0]
        isRight = value[1]
        if probability == 1.0:
            bin_list[size_of_bin - 1].append([probability, isRight])
            continue
        for i in range(0, size_of_bin):
            if (probability >= i * basic_width) & (probability < (i + 1) * basic_width):
                bin_list[i].append([probability, isRight])

    result_list = []
    for i in range(0, size_of_bin):
        value = bin_list[i]
        if len(value) == 0:
            result_list.append([None, None, 0])
            continue
        total_probability = 0
        total_right = 0
        for result in value:
            total_probability = total_probability + result[0]
            total_right = total_right + result[1]
        result_list.append([total_probability / len(value), total_right / (len(value)), len(value)])

    return result_list


def process_all(file_path, size_of_bin=10, dataset='atis', model='lstm-self-attention'):
    """

    :param file_path: the file_path is the path to your file.

    And the path must include file name.

    the file name is in this format: test_dataset_model.tsv.

    the file_path must in the format: /root/path/to/your/file/test_dataset.tsv

    The file must in this format:
    sentence\tground_truth\tpredict_label\tprobability\tright_or_not
    if prediction is right, right_or_not is assigned to 1, otherwise 0.

    :param size_of_bin: the numbers of how many bins

    :param dataset: the name of the dataset

    :param model: the name of the model

    :return:
    ece :the ece of this file
    dic :the details of the ECE information in json format
    """
    from collections import OrderedDict
    import json

    probability_list, right_or_not_list = get_probability_right_or_not(file_path)

    raw_list = get_raw_list(probability_list, right_or_not_list)

    bin_list = divide_into_bin(size_of_bin, raw_list)

    ece = calculate_ece(bin_list)
    dic = OrderedDict()
    dic['dataset-name'] = dataset
    dic['model-name'] = model
    dic['ECE'] = ece
    dic['details'] = []
    basic_width = 1 / size_of_bin
    for i in range(len(bin_list)):
        tem_dic = {}
        bin_name = str(i * basic_width) + '--' + str((i + 1) * basic_width)
        tem_dic[bin_name] = {'average_accuracy': bin_list[i][1], 'average_confidence': bin_list[i][0],
                             'samples_number_in_this_bin': bin_list[i][2]}
        dic['details'].append(tem_dic)

    return ece, json.dumps(dic)


ece, dic = process_all('/usr1/data/pliu3/ExplainaBoard/data/testPred/tc/cnn/test-CR.tsv',
                       size_of_bin=10, dataset='atis', model='lstm-self-attention')

print(dic)
print(ece)

