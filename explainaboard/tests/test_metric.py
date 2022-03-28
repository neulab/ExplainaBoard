import unittest

import sklearn.metrics

import explainaboard.metric


class TestMetric(unittest.TestCase):
    def test_accuracy(self):
        metric = explainaboard.metric.Accuracy()
        true = ['a', 'b', 'a', 'b', 'a', 'b']
        pred = ['a', 'b', 'a', 'b', 'b', 'a']
        result = metric.evaluate(true, pred, conf_value=0.05)
        self.assertAlmostEqual(result.value, 2.0 / 3.0)

    def test_f1_micro(self):
        metric = explainaboard.metric.F1Score(average='micro')
        true = ['a', 'b', 'a', 'b', 'a', 'a', 'c', 'c']
        pred = ['a', 'b', 'a', 'b', 'b', 'a', 'c', 'a']

        sklearn_f1 = sklearn.metrics.f1_score(true, pred, average='micro')
        result = metric.evaluate(true, pred, conf_value=0.05)
        self.assertAlmostEqual(result.value, sklearn_f1)

    def test_f1_macro(self):
        metric = explainaboard.metric.F1Score(average='macro')
        true = ['a', 'b', 'a', 'b', 'a', 'a', 'c', 'c']
        pred = ['a', 'b', 'a', 'b', 'b', 'a', 'c', 'a']
        sklearn_f1 = sklearn.metrics.f1_score(true, pred, average='macro')
        result = metric.evaluate(true, pred, conf_value=0.05)
        self.assertAlmostEqual(result.value, sklearn_f1)

    def test_hits(self):
        metric = explainaboard.metric.Hits()
        true = ['a', 'b', 'a', 'b', 'a', 'b']
        pred = [['a', 'b'], ['c', 'd'], ['c', 'a'], ['a', 'c'], ['b', 'a'], ['a', 'b']]
        result = metric.evaluate(true, pred, conf_value=0.05)
        self.assertAlmostEqual(result.value, 4.0 / 6.0)

    def test_mrr(self):
        metric = explainaboard.metric.MeanReciprocalRank()
        true = ['a', 'b', 'a', 'b', 'a', 'b']
        pred = [['a', 'b'], ['c', 'd'], ['c', 'a'], ['a', 'c'], ['b', 'a'], ['a', 'b']]
        result = metric.evaluate(true, pred, conf_value=0.05)
        self.assertAlmostEqual(result.value, 2.5 / 6.0)
