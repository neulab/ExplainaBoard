from dataclasses import asdict
import json
import os
import unittest

from integration_tests.utils import test_artifacts_path

from explainaboard.info import SysOutputInfo
from explainaboard.interpretation.bucket_interpretation import BucketIntpereter
from explainaboard.interpretation.combo_interpretation import ComboIntpereter
from explainaboard.interpretation.multi_bucket_interpretation import (
    MultiBucketIntpereter,
)


class InterpretationTest(unittest.TestCase):

    artifact_path = os.path.join(test_artifacts_path, "interpretations")
    report_file_path = os.path.join(artifact_path, "ner_report.json")

    with open(report_file_path, 'r') as file:
        report = SysOutputInfo.from_dict(json.loads(file.read()))
        feature_types = {}
        for analysis_obj in report.analyses:
            analysis = asdict(analysis_obj)
            if analysis["cls_name"] == "BucketAnalysis":
                feature_types[analysis["feature"]] = analysis["method"]

    def test_bucket_interpreter(self):

        analysis_example = self.report.results.analyses[7]
        interpreter = BucketIntpereter(analysis_example, self.feature_types).perform()

        # print(analysis_example.name)
        # print(interpreter.observations)
        # print(interpreter.suggestions)

        self.assertEqual(analysis_example.name, "span_length")
        self.assertEqual(
            interpreter.observations["F1"][0].keywords, "performance_description"
        )
        self.assertEqual(
            interpreter.suggestions["F1"][0].keywords, "correlation_description"
        )
        self.assertEqual(
            interpreter.suggestions["F1"][0].content,
            "If the absolute value of correlation is greater than 0.9,"
            " it means that the performance of the system is highly"
            " affected by features."
            " Consider improving the training samples under appropriate"
            " feature value of span_length to improve"
            " the model performance.",
        )

    def test_combo_interpreter(self):

        analysis_example = self.report.results.analyses[1]
        interpreter = ComboIntpereter(analysis_example).perform()

        # print(analysis_example.name)
        # print(interpreter.observations)
        # print(interpreter.suggestions)

        self.assertEqual(
            analysis_example.name, "combo(span_true_label,span_pred_label)"
        )
        self.assertEqual(
            interpreter.observations[0].keywords, "misprediction_description"
        )
        self.assertEqual(
            interpreter.suggestions[0].keywords, "misprediction_description"
        )
        self.assertEqual(
            interpreter.suggestions[0].content,
            "These samples, which are frequently"
            " mispredicted by the model, need to be"
            " prioritized for solutions.",
        )

    def test_multi_bucket_interpreter(self):

        analysis_example = self.report.results.analyses
        interpreter = MultiBucketIntpereter(
            analysis_example, self.feature_types
        ).perform()

        # print(interpreter.observations)
        # print(interpreter.suggestions)

        self.assertEqual(
            interpreter.observations["F1"][0].keywords, "salient_feature_description"
        )
        self.assertEqual(
            interpreter.suggestions["F1"][0].keywords, "salient_feature_description"
        )
        self.assertEqual(
            interpreter.suggestions["F1"][0].content,
            "The performance of the system is highly affected by"
            " these features. Consider augment the training samples"
            " to improve the model performance.",
        )
