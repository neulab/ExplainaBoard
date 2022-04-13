from __future__ import annotations

import unittest

from explainaboard.utils.typing_utils import narrow


class TestTypingUtils(unittest.TestCase):
    def test_narrow(self):
        a: str | int = 's'
        self.assertEqual(narrow(a, str), a)
        self.assertRaises(TypeError, lambda: narrow(a, int))
