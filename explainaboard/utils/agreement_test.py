"""Tests for agreement.py."""

from __future__ import annotations

import unittest

import numpy as np

from explainaboard.utils.agreement import fleiss_kappa


class FleissKappaTest(unittest.TestCase):
    def test_FleissKappa(self):
        A = np.array(
            [
                [0, 0, 0, 0, 14],
                [0, 2, 6, 4, 2],
                [0, 0, 3, 5, 6],
                [0, 3, 9, 2, 0],
                [2, 2, 8, 1, 1],
                [7, 7, 0, 0, 0],
                [3, 2, 6, 3, 0],
                [2, 5, 3, 2, 2],
                [6, 5, 2, 1, 0],
                [0, 2, 2, 3, 7],
            ]
        )

        self.assertEqual(fleiss_kappa(A), 0.2099)
