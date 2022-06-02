from __future__ import annotations

import os
import unittest

from eaas import Config
from eaas.async_client import AsyncClient
import numpy as np
import sklearn.metrics

from explainaboard import FileType, get_processor, Source, TaskType
from explainaboard.loaders import get_custom_dataset_loader
import explainaboard.metric
from explainaboard.tests.utils import test_artifacts_path


class TestMetric(unittest.TestCase):
    def test_accuracy(self):
        metric = explainaboard.metric.AccuracyConfig(name='Accuracy').to_metric()
        true = ['a', 'b', 'a', 'b', 'a', 'b']
        pred = ['a', 'b', 'a', 'b', 'b', 'a']
        result = metric.evaluate(true, pred, conf_value=0.05)
        self.assertAlmostEqual(result.value, 2.0 / 3.0)

    def test_correct_score(self):
        metric = explainaboard.metric.CorrectCountConfig(
            name='CorrectCount'
        ).to_metric()
        true = ['a', 'b', 'a', 'b', 'a', 'b']
        pred = ['a', 'b', 'a', 'b', 'b', 'a']
        result = metric.evaluate(true, pred, conf_value=0.05)
        self.assertAlmostEqual(result.value, 4)

    def test_seq_correct_score(self):
        metric = explainaboard.metric.SeqCorrectCountConfig(
            name='SeqCorrectCount'
        ).to_metric()
        true = [
            {
                "start_idx": [8, 17, 39, 46, 58, 65, 65, 80],
                "end_idx": [8, 18, 40, 47, 59, 65, 66, 81],
                "corrections": [
                    ["the"],
                    ["found"],
                    ["other"],
                    ["there"],
                    ["chickens."],
                    ["in"],
                    ["which"],
                    ["selling"],
                ],
            }
        ]
        pred = [
            {
                "start_idx": [8, 17, 39, 46, 58],
                "end_idx": [8, 18, 40, 47, 59],
                "corrections": [
                    ["the"],
                    ["found"],
                    ["other"],
                    ["there"],
                    ["chickens."],
                ],
            }
        ]
        result = metric.evaluate(true, pred)
        self.assertAlmostEqual(result.value, 5)

    def test_f1_micro(self):
        metric = explainaboard.metric.F1ScoreConfig(
            name='F1', average='micro'
        ).to_metric()
        true = ['a', 'b', 'a', 'b', 'a', 'a', 'c', 'c']
        pred = ['a', 'b', 'a', 'b', 'b', 'a', 'c', 'a']

        sklearn_f1 = sklearn.metrics.f1_score(true, pred, average='micro')
        result = metric.evaluate(true, pred, conf_value=0.05)
        self.assertAlmostEqual(result.value, sklearn_f1)

    def test_f1_macro(self):
        metric = explainaboard.metric.F1ScoreConfig(
            name='F1', average='macro'
        ).to_metric()
        true = ['a', 'b', 'a', 'b', 'a', 'a', 'c', 'c']
        pred = ['a', 'b', 'a', 'b', 'b', 'a', 'c', 'a']
        sklearn_f1 = sklearn.metrics.f1_score(true, pred, average='macro')
        result = metric.evaluate(true, pred, conf_value=None)
        self.assertAlmostEqual(result.value, sklearn_f1)

    def test_hits(self):
        metric = explainaboard.metric.HitsConfig(name='Hits').to_metric()
        true = ['a', 'b', 'a', 'b', 'a', 'b']
        pred = [['a', 'b'], ['c', 'd'], ['c', 'a'], ['a', 'c'], ['b', 'a'], ['a', 'b']]
        result = metric.evaluate(true, pred, conf_value=0.05)
        self.assertAlmostEqual(result.value, 4.0 / 6.0)

    def test_mrr(self):
        metric = explainaboard.metric.MeanReciprocalRankConfig(name='MRR').to_metric()
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

        metric = explainaboard.metric.SeqF1ScoreConfig(
            name='MicroF1', average='micro', tag_schema='bio'
        ).to_metric()
        result = metric.evaluate(true, pred, conf_value=None)
        self.assertAlmostEqual(result.value, 2.0 / 3.0)

        metric = explainaboard.metric.SeqF1ScoreConfig(
            name='MacroF1', average='macro', tag_schema='bio'
        ).to_metric()
        result = metric.evaluate(true, pred, conf_value=None)
        self.assertAlmostEqual(result.value, 3.0 / 4.0)

    def _get_eaas_request(
        self,
        sys_output: list[dict],
        metric_names: list[str],
        eaas_client: AsyncClient,
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
            inputs, metrics=metric_names.copy(), calculate=['corpus', 'stats']
        )

    def test_eaas_decomposabiltiy(self):
        # Get data
        tsv_dataset = os.path.join(
            test_artifacts_path, "machine_translation", "dataset.tsv"
        )
        txt_output = os.path.join(
            test_artifacts_path, "machine_translation", "output.txt"
        )
        loader = get_custom_dataset_loader(
            TaskType.machine_translation,
            tsv_dataset,
            txt_output,
            Source.local_filesystem,
            Source.local_filesystem,
            FileType.tsv,
            FileType.text,
        )
        sys_output = loader.load()

        # Initialize client and decide which metrics to test
        eaas_client = AsyncClient(Config())
        metric_names = ['rouge1', 'bleu', 'chrf', 'length_ratio']
        # Uncomment the following line to test all metrics,
        # but beware that it will be very slow
        # metric_names = eaas_client._valid_metrics
        metrics = [
            explainaboard.metric.EaaSMetricConfig(name=name).to_metric()
            for name in metric_names
        ]

        # Get results for full data and half data
        half_bound = int(len(sys_output) / 2)
        full_request = self._get_eaas_request(sys_output, metric_names, eaas_client)
        half_request = self._get_eaas_request(
            sys_output[:half_bound], metric_names, eaas_client
        )
        half_ids = np.array(range(half_bound))
        full_result = full_request.get_result()
        half_result = half_request.get_result()

        for i, (name, metric) in enumerate(zip(metric_names, metrics)):
            with self.subTest(msg=name):
                full_stats = explainaboard.metric.EaaSMetricStats(
                    name=name, pos=i, eaas_request=full_request
                )
                half_stats = explainaboard.metric.EaaSMetricStats(
                    name=name, pos=i, eaas_request=half_request
                )
                split_stats = full_stats.filter(half_ids)
                # EaaS-returned value should be same as explainaboard-calculated value
                self.assertAlmostEqual(
                    full_result['scores'][i]['corpus'],
                    metric.evaluate_from_stats(full_stats).value,
                )
                self.assertAlmostEqual(
                    half_result['scores'][i]['corpus'],
                    metric.evaluate_from_stats(half_stats).value,
                )
                # Stats calculated over half corpus should be the same as the stats
                # split away from the full corpus
                self.assertAlmostEqual(
                    metric.evaluate_from_stats(half_stats).value,
                    metric.evaluate_from_stats(split_stats).value,
                )

    def test_qa_metrics(self):
        json_en_dataset = os.path.join(
            test_artifacts_path, "extractive_qa", "dataset-xquad-en.json"
        )
        json_en_output = os.path.join(
            test_artifacts_path, "extractive_qa", "output-xquad-en.json"
        )
        loader = get_custom_dataset_loader(
            TaskType.qa_extractive,
            json_en_dataset,
            json_en_output,
            Source.local_filesystem,
            Source.local_filesystem,
            FileType.json,
            FileType.json,
        )
        data = loader.load()

        metadata = {
            "task_name": TaskType.qa_extractive.value,
            "dataset_name": "squad",
            "metric_names": ["F1ScoreQA", "ExactMatchQA"],
        }

        processor = get_processor(TaskType.qa_extractive)

        sys_info = processor.process(metadata, data)

        # analysis.write_to_directory("./")
        self.assertIsNotNone(sys_info.results.fine_grained)
        self.assertGreater(len(sys_info.results.overall), 0)
        self.assertAlmostEqual(
            sys_info.results.overall["ExactMatch"].value,
            0.6974789915966386,
            2,
            "almost equal",
        )
        # should be 0.8235975260931867
        self.assertAlmostEqual(
            sys_info.results.overall["F1"].value,
            0.8235975260931867,
            2,
            "almost equal",
        )
