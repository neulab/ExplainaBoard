from __future__ import annotations

import os
import unittest

from eaas import Config
from eaas.async_client import AsyncClient
from integration_tests.utils import test_artifacts_path
import numpy as np

from explainaboard import FileType, get_processor_class, Source, TaskType
from explainaboard.loaders import get_loader_class
import explainaboard.metrics.accuracy
import explainaboard.metrics.eaas
import explainaboard.metrics.f1_score
from explainaboard.metrics.metric import Score
import explainaboard.metrics.ranking
import explainaboard.metrics.text_to_sql


class MetricTest(unittest.TestCase):
    def _get_eaas_request(
        self,
        sys_output: list[dict],
        metric_names: list[str],
        eaas_client: AsyncClient,
    ) -> explainaboard.metrics.eaas.AsyncRequest:
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
            inputs, metrics=metric_names.copy(), calculate=["corpus", "stats"]
        )

    def test_eaas_decomposabiltiy(self) -> None:
        # Get data
        tsv_dataset = os.path.join(
            test_artifacts_path, "machine_translation", "dataset.tsv"
        )
        txt_output = os.path.join(
            test_artifacts_path, "machine_translation", "output.txt"
        )
        loader = get_loader_class(TaskType.machine_translation)(
            tsv_dataset,
            txt_output,
            Source.local_filesystem,
            Source.local_filesystem,
            FileType.tsv,
            FileType.text,
        )
        sys_output = loader.load().samples

        # Initialize client and decide which metrics to test
        eaas_client = AsyncClient(Config())
        metric_names = ["rouge1", "bleu", "chrf", "length_ratio"]
        # Uncomment the following line to test all metrics,
        # but beware that it will be very slow
        # metric_names = eaas_client._valid_metrics
        metrics = [
            explainaboard.metrics.eaas.EaaSMetricConfig(name=name).to_metric()
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
                full_stats = explainaboard.metrics.eaas.EaaSMetricStats(
                    name=name, pos=i, eaas_request=full_request
                )
                half_stats = explainaboard.metrics.eaas.EaaSMetricStats(
                    name=name, pos=i, eaas_request=half_request
                )
                split_stats = full_stats.filter(half_ids)
                # EaaS-returned value should be same as explainaboard-calculated value
                self.assertAlmostEqual(
                    full_result["scores"][i]["corpus"],
                    metric.evaluate_from_stats(full_stats)
                    .get_value(Score, "score")
                    .value,
                )
                self.assertAlmostEqual(
                    half_result["scores"][i]["corpus"],
                    metric.evaluate_from_stats(half_stats)
                    .get_value(Score, "score")
                    .value,
                )
                # Stats calculated over half corpus should be the same as the stats
                # split away from the full corpus
                self.assertAlmostEqual(
                    metric.evaluate_from_stats(half_stats)
                    .get_value(Score, "score")
                    .value,
                    metric.evaluate_from_stats(split_stats)
                    .get_value(Score, "score")
                    .value,
                )

    def test_qa_metrics(self) -> None:
        json_en_dataset = os.path.join(
            test_artifacts_path, "extractive_qa", "dataset-xquad-en.json"
        )
        json_en_output = os.path.join(
            test_artifacts_path, "extractive_qa", "output-xquad-en.json"
        )
        loader = get_loader_class(TaskType.qa_extractive)(
            json_en_dataset,
            json_en_output,
            Source.local_filesystem,
            Source.local_filesystem,
            FileType.json,
            FileType.json,
        )
        data = loader.load().samples

        metadata = {
            "task_name": TaskType.qa_extractive.value,
            "dataset_name": "squad",
            "metric_names": ["F1ScoreQA", "ExactMatchQA"],
            "source_language": "en",
        }

        processor = get_processor_class(TaskType.qa_extractive)()

        sys_info = processor.process(metadata, data)

        self.assertGreater(len(sys_info.results.analyses), 0)
        overall = sys_info.results.overall["example"]
        self.assertGreater(len(overall), 0)
        self.assertAlmostEqual(
            overall["ExactMatch"].get_value(Score, "score").value,
            0.6974789915966386,
            2,
        )
        self.assertAlmostEqual(
            overall["F1"].get_value(Score, "score").value, 0.8235975260931867, 2
        )

    def test_sql_exactsetmatch(self):
        metric = explainaboard.metrics.text_to_sql.SQLExactSetMatchConfig(
            db_dir="https://storage.googleapis.com/inspired-public-data/"
            "explainaboard/task_data/text_to_sql/database",
            table_path="https://storage.googleapis.com/inspired-public-data/"
            "explainaboard/task_data/text_to_sql/tables/tables.json",
        ).to_metric()
        true = [
            ["select distinct country from singer where age > 20", "concert_singer"],
            ["select distinct country from singer where age > 20", "concert_singer"],
            ["select distinct country from singer where age > 20", "concert_singer"],
        ]
        pred = [
            ["select distinct country from singer where age > 20", "concert_singer"],
            ["select distinct country from singer where age > 25", "concert_singer"],
            ["select distinct country from singer where age = 20", "concert_singer"],
        ]
        result = metric.evaluate(true, pred)
        self.assertAlmostEqual(result.get_value(Score, "score").value, 2.0 / 3.0)

    def test_sql_execution(self):
        metric = explainaboard.metrics.text_to_sql.SQLExecutionConfig(
            db_dir="https://storage.googleapis.com/inspired-public-data/"
            "explainaboard/task_data/text_to_sql/database",
            table_path="https://storage.googleapis.com/inspired-public-data/"
            "explainaboard/task_data/text_to_sql/tables/tables.json",
        ).to_metric()
        true = [
            ["select distinct country from singer where age > 20", "concert_singer"],
            ["select distinct country from singer where age > 20", "concert_singer"],
            ["select distinct country from singer where age > 20", "concert_singer"],
        ]
        pred = [
            ["select distinct country from singer where age > 20", "concert_singer"],
            ["select distinct country from singer where age > 25", "concert_singer"],
            ["select distinct country from singer where age = 20", "concert_singer"],
        ]
        result = metric.evaluate(true, pred)
        self.assertAlmostEqual(result.get_value(Score, "score").value, 1.0 / 3.0)
