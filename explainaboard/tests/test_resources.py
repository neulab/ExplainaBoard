from __future__ import annotations

import unittest

from explainaboard.utils.load_resources import get_customized_features


class TestResources(unittest.TestCase):
    def test_get_customized_features(self):

        self.assertEqual(
            get_customized_features()["sst2"],
            {
                'label': {
                    'dtype': 'string',
                    'description': 'the true label',
                    'num_buckets': 2,
                }
            },
        )
