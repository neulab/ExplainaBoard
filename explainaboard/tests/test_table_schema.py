import unittest

from explainaboard import TaskType
from explainaboard.table_schema import table_schemas


class TestTableSchema(unittest.TestCase):
    def test_table_schemas(self):

        self.assertEqual(len(table_schemas[TaskType.text_classification]), 3)

        self.assertEqual(len(table_schemas[TaskType.summarization]), 3)

        self.assertEqual(len(table_schemas[TaskType.question_answering_extractive]), 4)

        self.assertEqual(len(table_schemas[TaskType.named_entity_recognition]), 4)


if __name__ == '__main__':
    unittest.main()
