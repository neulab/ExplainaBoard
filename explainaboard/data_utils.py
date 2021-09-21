import pandas as pd
import numpy as np


def get_probability_right_or_not(file_path, prob_col, right_or_not_col=None, answer_cols=None):
    """

    :param file_path: the file_path is the path to your file.

    And the path must include file name.

    the file name is in this format: test_dataset_model.tsv.
    the file_path must in the format: /root/path/to/your/file/test_dataset.tsv

    :param prob_col: the index of the column containing the probability output by the model
    :param right_or_not_col: the index of the column indicating whether the answer is right
    :param answer_cols: the indices of two columns containing the true and system-output answer

    Either right_or_not_col or answer_cols must be populated
    """

    result = pd.read_csv(file_path, sep='\t', header=None)

    probability_list = np.array(result[prob_col]).tolist()
    if right_or_not_col is not None:
        right_or_not_list = np.array(result[right_or_not_col]).tolist()
    elif answer_cols is not None:
        right_or_not_list = np.array(result[answer_cols[0]].eq(result[answer_cols[1]])).tolist()
    else:
        raise ValueError('right_or_not_cols or answer_cols must not be None')

    return probability_list, right_or_not_list


def tsv_to_lists(path_file, col_ids, fail_on_short_line=True):
    """
    Grab a list of columns from a tsv file
    :param path_file: The path to the file
    :param col_ids: The integer column IDs
    :param fail_on_short_line: Whether to fail if there is a line that's too short
    :return:
    """
    ret_lists = tuple([] for _ in col_ids)
    max_col = max(col_ids)
    with open(path_file, "r") as fin:
        for line in fin:
            line = line.rstrip("\n")
            cols = line.split("\t")
            if max_col < len(cols):
                for col_id, col_list in zip(col_ids, ret_lists):
                    col_list.append(cols[col_id])
            elif fail_on_short_line:
                raise ValueError(f'Illegal short line in {path_file}\n{line}')
    return ret_lists
