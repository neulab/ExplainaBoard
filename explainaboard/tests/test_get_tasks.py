import unittest
from explainaboard import get_task_categories

class TestTasks(unittest.TestCase):
    def test_get_task_categories(self):
        task_categories = get_task_categories()
        self.assertTrue(isinstance(task_categories, list))

