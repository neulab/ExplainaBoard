from __future__ import annotations

import os
import pathlib
import unittest

from eaas import Config
import numpy as np
import sklearn.metrics

from explainaboard import FileType, Source, TaskType
from explainaboard.loaders.loader import get_loader
import explainaboard.metric
from explainaboard.utils.async_eaas import AsyncEaaSClient

artifacts_path = os.path.dirname(pathlib.Path(__file__)) + "/artifacts/"


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

    def test_ner_f1(self):

        true = [
            ['O', 'O', 'B-MISC', 'I-MISC', 'B-MISC', 'O', 'O'],
            ['B-PER', 'I-PER', 'O'],
        ]
        pred = [
            ['O', 'O', 'B-MISC', 'I-MISC', 'B-MISC', 'I-MISC', 'O'],
            ['B-PER', 'I-PER', 'O'],
        ]

        metric = explainaboard.metric.BIOF1Score(average='micro')
        result = metric.evaluate(true, pred, conf_value=0.05)
        self.assertAlmostEqual(result.value, 2.0 / 3.0)

        metric = explainaboard.metric.BIOF1Score(average='macro')
        result = metric.evaluate(true, pred, conf_value=0.05)
        self.assertAlmostEqual(result.value, 3.0 / 4.0)

    def _get_eaas_request(
        self,
        sys_output: list[dict],
        metric_names: list[str],
        eaas_client: AsyncEaaSClient,
    ):
        # Queue up EaaS client request for all metrics
        inputs = []
        for feature_table in sys_output:
            inputs.append(
                {
                    "source": feature_table["source"],
                    "references": [feature_table["reference"]],
                    "hypothesis": feature_table["hypothesis"],
                }
            )
        return eaas_client.async_score(
            inputs,
            task="sum",  # TODO(pengfei): this should be generalized
            metrics=metric_names.copy(),
            lang="en",
            cal_attributes=False,
        )

    def test_eaas_decomposabiltiy(self):

        # Get data
        path_data = artifacts_path + "test-mt.tsv"
        loader = get_loader(
            TaskType.machine_translation,
            path_data,
            Source.local_filesystem,
            FileType.tsv,
        )
        sys_output = list(loader.load())

        # Initialize client and decide which metrics to test
        eaas_client = AsyncEaaSClient(Config())
        metric_names = ['rouge1', 'bleu', 'chrf']
        # Uncomment the following line to test all metrics,
        # but beware that it will be very slow
        # metric_names = eaas_client._valid_metrics
        metrics = [explainaboard.metric.EaaSMetric(name=name) for name in metric_names]

        # Get results for full data and half data
        half_bound = int(len(sys_output) / 2)
        full_request = self._get_eaas_request(sys_output, metric_names, eaas_client)
        half_request = self._get_eaas_request(
            sys_output[:half_bound], metric_names, eaas_client
        )
        half_ids = np.array(range(half_bound))
        full_result = full_request.get_result()
        half_result = half_request.get_result()

        for name, metric in zip(metric_names, metrics):
            with self.subTest(msg=name):
                full_stats = explainaboard.metric.EaaSMetricStats(
                    name=name, eaas_request=full_request
                )
                half_stats = explainaboard.metric.EaaSMetricStats(
                    name=name, eaas_request=half_request
                )
                split_stats = full_stats.filter(half_ids)
                # EaaS-returned value should be same as explainaboard-calculated value
                self.assertAlmostEqual(
                    full_result['corpus_level'][f'corpus_{name}'],
                    metric.evaluate_from_stats(full_stats).value,
                )
                self.assertAlmostEqual(
                    half_result['corpus_level'][f'corpus_{name}'],
                    metric.evaluate_from_stats(half_stats).value,
                )
                # Stats calculated over half corpus should be the same as the stats
                # split away from the full corpus
                self.assertAlmostEqual(
                    metric.evaluate_from_stats(half_stats).value,
                    metric.evaluate_from_stats(split_stats).value,
                )
