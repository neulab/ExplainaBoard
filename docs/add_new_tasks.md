# Add New Tasks

Let's take `text_classification` as an example to show how to add a new task for ExplainaBoard.

To do so, you would first need to add your task to the modules `tasks.py` and `table_schema.py`. 
After doing so, you would also need to create a loader, builder, processor and unittest for the
new task, placed under the relevant directories.


## Task and Format Declaration
(1) All the supported tasks are listed in **tasks.py**. If your task name is not listed in the file,
please add your task to `TaskType` (enum) and the task list `_task_categories`. Task names can not 
contain `space` and different words should be connected using `-`.

(2) If the format of your task's dataset is not covered by `FileType` in the file:
`explainaboard/constants.py`, please manually add the new format.

For example:
```python
class TaskType(str, Enum):
    text_classification = "text-classification"


_task_categories: List[TaskCategory] = [
    TaskCategory("text-classification", "predicting a class index or boolean value",
                 [Task(TaskType.text_classification, True, ["F1score", "Accuracy"])]),
]
```
where the parameters in `TaskCategory` refers to the task's name, description and the list of tasks


## Table Schema Declaration
(1) Add a new table schema for your task in **table_schema.py**. 
Each schema is a list of dictionary, which is used to instruct how to print bucket-level cases 
in the frontend table (the number of list denotes the number of table columns)
Currently, the table schema is characterized by:
* field_key:str:  this is used to retrieve data from system output file
* sort:bool: whether this column is sortable
* filter:bool: whether this column is filterable
* label:str: the text to be printed of this column in the table head

The blocks you add should match the table columns in the system output file,
for example:
```python
table_schemas[TaskType.text_classification] = [
    {
        "field_key": "text",
        "sort": False,
        "filter": False,
        "label": "Text"
    },
    {
        "field_key": "true_label",
        "sort": True,
        "filter": True,
        "label": "True Label"
    },
    {
        "field_key": "predicted_label",
        "sort": True,
        "filter": True,
        "label": "Prediction"
    },
]
```


## Create a Loader module for your task

(1) Create a new python file `text_classification.py` in the folder `explainaboard/loaders/`

(2) In this file, we need to:
* create a data loader for text classification task inheriting from the class `Loader`
* re-implement the member function `def load(self)`

Specifically:
  
```python
from typing import Dict, Iterable, List
from explainaboard.constants import *
from .loader import register_loader
from .loader import Loader


@register_loader(TaskType.text_classification) # register this laod function
class TextClassificationLoader(Loader):
    """
    Validate and Reformat system output file with tsv format:
    text \t true_label \t predicted_label
    usage:
        please refer to `test_loaders.py`
    """

    def load(self) -> Iterable[Dict]:
        """
        :param path_system_output: the path of system output file with following format:
        text \t label \t predicted_label
        :return: class object
        """
        raw_data = self._load_raw_data_points()
        data: List[Dict] = []
        if self._file_type == FileType.tsv:
            for id, dp in enumerate(raw_data):
                text, true_label, predicted_label = dp[:3]
                data.append({"id": id,
                             "text": text.strip(),
                             "true_label": true_label.strip(),
                             "predicted_label": predicted_label.strip()})
        else:
            raise NotImplementedError
        return data
```

(3) Import this module (text_classification.py) in `__init__.py`
For example, in this file `explainaboard/loaders/__init__.py`, we have:
```python
from . import text_classification
```


## Create a Builder module for your task
(1) Create a new python file, `text_classification.py` in the folder: `explainaboard/builders/`:
Implement it



## Create a Processor module for your task

(1) Create a new python file `text_classification.py` in the folder `explainaboard/processors/`

(2) In this file, we need to:
* create a processor for text classification task inheriting from the class `Processor`
* define features that you aim to use for this task
* re-implement constructor function

(3) Import this module (text_classification.py) in `__init__.py`
For example, in this file `explainaboard/processors/__init__.py`, we have: 
```python
from . import text_classification
```

## Finally, create a Unittest module for your task

(1) Create a new python file `test_text_classification.py` in the folder `explainaboard/tests/`

(2) Implement a unittest for this task

