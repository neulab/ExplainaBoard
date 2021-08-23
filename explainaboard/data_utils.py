import pandas as pd
import numpy as np


def get_probability_right_or_not(file_path, prob_col, right_or_not_col):
    """

    :param file_path: the file_path is the path to your file.

    And the path must include file name.

    the file name is in this format: test_dataset_model.tsv.
    the file_path must in the format: /root/path/to/your/file/test_dataset.tsv

    prob_col and right_or_not_col are the columnwise indices of the probability and whether it's right or not
    if prediction is right, right_or_not is assigned to 1, otherwise 0.

    """

    result = pd.read_csv(file_path, sep='\t', header=None)

    probability_list = np.array(result[prob_col]).tolist()
    right_or_not_list = np.array(result[right_or_not_col]).tolist()

    return probability_list, right_or_not_list
