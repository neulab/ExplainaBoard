from __future__ import annotations

import unittest

from explainaboard.utils.load_resources import get_custmomized_features


class TestResources(unittest.TestCase):
    def test_get_customized_features(self):

        print(get_custmomized_features())

        self.assertEqual(
            get_custmomized_features()["sst2"],
            {
                'label': {
                    'dtype': 'string',
                    'description': 'the true label',
                    'num_buckets': 2,
                }
            },
        )
