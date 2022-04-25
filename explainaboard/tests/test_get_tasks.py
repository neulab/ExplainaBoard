import unittest

from explainaboard import get_task_categories


class TestTasks(unittest.TestCase):
    def test_get_task_categories(self):
        task_categories = get_task_categories()
        self.assertTrue(isinstance(task_categories, list))
        task_category_names = [category.name for category in task_categories]
        self.assertEqual(
            len(task_category_names),
            len(set(task_category_names)),
            "task category names should be unique",
        )
        for task_category in task_categories:
            self.assertIsNotNone(task_category.description)
            self.assertIsNotNone(task_category.name)
            for task in task_category.tasks:
                self.assertEqual(
                    len(task.supported_metrics),
                    len(set(task.supported_metrics)),
                    f"duplicate metrics in {task.name}",
                )
                self.assertGreater(len(task.supported_formats.custom_dataset), 0)
                self.assertGreater(len(task.supported_formats.system_output), 0)
