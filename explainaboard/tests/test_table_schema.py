import unittest

from explainaboard import TaskType
from explainaboard.table_schema import table_schemas


class TestTableSchema(unittest.TestCase):
    def test_table_schemas(self):

        print(table_schemas[TaskType.text_classification])

        self.assertEqual(len(table_schemas[TaskType.text_classification]), 3)

        print(table_schemas[TaskType.summarization])
        self.assertEqual(len(table_schemas[TaskType.summarization]), 3)

        print(table_schemas[TaskType.question_answering_extractive])
        self.assertEqual(len(table_schemas[TaskType.question_answering_extractive]), 4)

        print(table_schemas[TaskType.named_entity_recognition])
        self.assertEqual(len(table_schemas[TaskType.named_entity_recognition]), 4)


if __name__ == '__main__':
    unittest.main()
