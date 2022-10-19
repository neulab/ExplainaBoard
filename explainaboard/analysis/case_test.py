"""Tests for explainaboard.analysis.case."""

from __future__ import annotations

import unittest

from explainaboard.analysis.case import AnalysisCaseCollection


class AnalysisCaseCollectionTest(unittest.TestCase):
    def test_values(self):
        # These invocation does not raise anything.
        case1 = AnalysisCaseCollection(samples=[1], interval=[1.0, 2.0])
        self.assertEqual(case1.samples, [1])
        self.assertEqual(case1.interval, [1.0, 2.0])
        self.assertIsNone(case1.name)

        case2 = AnalysisCaseCollection(samples=[1], name="test")
        self.assertEqual(case2.samples, [1])
        self.assertIsNone(case2.interval)
        self.assertEqual(case2.name, "test")

        with self.assertRaisesRegex(ValueError, r"^Either"):
            AnalysisCaseCollection(samples=[1])

        with self.assertRaisesRegex(ValueError, r"^Both"):
            AnalysisCaseCollection(samples=[1], interval=[1.0, 2.0], name="test")
