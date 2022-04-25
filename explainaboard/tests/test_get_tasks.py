import unittest

from explainaboard import get_task_categories
from explainaboard.loaders.loader_registry import get_supported_file_types_for_loader
from explainaboard.processors.processor_registry import get_metric_list_for_processor


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
                supported_metrics = get_metric_list_for_processor(task.name)
                supported_formats = get_supported_file_types_for_loader(task.name)
                self.assertEqual(
                    len(supported_metrics),
                    len(set([x.name for x in supported_metrics])),
                    f"duplicate metric names in {task.name}",
                )
                self.assertGreater(len(supported_formats.custom_dataset), 0)
                self.assertGreater(len(supported_formats.system_output), 0)
