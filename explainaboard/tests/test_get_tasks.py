import unittest
from explainaboard import get_task_categories


class TestTasks(unittest.TestCase):
    def test_get_task_categories(self):
        task_categories = get_task_categories()
        self.assertTrue(isinstance(task_categories, list))
        for task_category in task_categories:
            self.assertIsNotNone(task_category.description)
            self.assertIsNotNone(task_category.name)
            for task in task_category.tasks:
                self.assertEqual(
                    len(task.supported_metrics),
                    len(set(task.supported_metrics)),
                    f"duplicate metrics in {task.name}",
                )
