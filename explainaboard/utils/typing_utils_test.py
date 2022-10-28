from __future__ import annotations

import unittest

from explainaboard.utils.typing_utils import narrow


class TypingUtilsTest(unittest.TestCase):
    def test_narrow(self):
        a: str | int = "s"
        self.assertEqual(narrow(str, a), a)
        self.assertRaises(TypeError, lambda: narrow(int, a))
