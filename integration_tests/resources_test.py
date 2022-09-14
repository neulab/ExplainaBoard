from __future__ import annotations

import unittest

from explainaboard.utils.load_resources import get_customized_features


class ResourcesTest(unittest.TestCase):
    def test_get_customized_features(self):

        self.assertEqual(
            {
                "custom_features": {
                    "example": {
                        "label": {
                            "cls_name": "Value",
                            "dtype": "string",
                            "description": "the true label",
                        }
                    }
                },
                "custom_analyses": [
                    {
                        "cls_name": "BucketAnalysis",
                        "feature": "label",
                        "level": "example",
                        "num_buckets": 2,
                        "method": "discrete",
                    }
                ],
            },
            get_customized_features()["sst2"],
        )
