"""Utility functions to calculate agreement scores."""

from __future__ import annotations

import numpy as np


def fleiss_kappa(M: np.ndarray) -> float:
    """Computes Fleiss' kappa for group of annotators.

    This is based on:
    https://towardsdatascience.com/inter-annotator-agreement-2f46c6d37bf3

    Args:
        M: a matrix of shape (:attr:'N', :attr:'k') with 'N' = number
        of subjects and 'k' = the number of categories.
            'M[i, j]' represent the number of raters who assigned the 'i'th
            subject to the 'j'th category.

    Returns:
         Fleiss' kappa score
    """
    N, k = M.shape  # N is # of items, k is # of categories
    n_annotators = float(np.sum(M[0, :]))  # # of annotators
    tot_annotations = N * n_annotators  # the total # of annotations
    category_sum = np.sum(M, axis=0)  # the sum of each category over all items

    # chance agreement
    p = category_sum / tot_annotations  # the distribution of each category over
    # all annotations
    PbarE = np.sum(p * p)  # average chance agreement over all categories

    # observed agreement
    P = (np.sum(M * M, axis=1) - n_annotators) / (n_annotators * (n_annotators - 1))
    Pbar = np.sum(P) / N  # add all observed agreement chances per item and divide
    # by amount of items

    return round((Pbar - PbarE) / (1 - PbarE), 4)
