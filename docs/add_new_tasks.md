# Add New Tasks

Let's take `text_classification` as an example to show how to add a new task for ExplainaBoard.

To do so, you would first need to add your task to the modules `tasks.py`. 
After doing so, you would also need to create a `Loader`, `Processor`, and unit tests for the
new task, placed under the relevant directories.


## Task and Format Declaration
(1) All the supported tasks are listed in **tasks.py**. If your task name is not listed in the file,
please add your task to `TaskType` (enum) and the task list `_task_categories`. Task names can not 
contain `space` and different words should be connected using `-`.

(2) If the format of your task's dataset is not covered by `FileType` in the file
`explainaboard/constants.py`, please manually add the new format (in the case that your
task uses a standard format such as tsv, it is not necessary to add a new type).

For example:
```python
class TaskType(str, Enum):
    text_classification = "text-classification"


_task_categories: List[TaskCategory] = [
    TaskCategory("text-classification", "predicting a class index or boolean value",
                 [Task(TaskType.text_classification, True, ["F1score", "Accuracy"])]),
]
```
where the parameters in `TaskCategory` refers to the task's name, description, and the list of tasks

## Create a Loader module for your task

(1) Create a new python file `text_classification.py` in the folder `explainaboard/loaders/`

(2) In this file, we need to:
* create a data loader for text classification task inheriting from the class `Loader`
* implement the member function `def load(self)`

Here is the example for text classification:
  
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
        super().load()
        data: List[Dict] = []
        if self._file_type == FileType.tsv:
            for id, dp in enumerate(self._raw_data):
                text, true_label, predicted_label = dp[:3]
                data.append(
                    {
                        "id": str(id),
                        "text": text.strip(),
                        "true_label": true_label.strip(),
                        "predicted_label": predicted_label.strip(),
                    }
                )
        else:
            raise NotImplementedError
        return data
```

In general, it is a good idea to look at the loader for the most similar variety of
task to the one you're trying to implement to get hints.

(3) Import this module (`text_classification.py`) in `__init__.py`
For example, in this file `explainaboard/loaders/__init__.py`, we have:
```python
from . import text_classification
```

## Create a Processor module for your task

(1) Create a new python file `text_classification.py` in the folder `explainaboard/processors/`

(2) In this file, we need to:
* create a processor for text classification task inheriting from the class `Processor`
* define features that you aim to use for this task in the `_features` variable
* [implement the features](add_new_features.md) that you want to use to perform analysis

Note that for simpler tests, this is mostly all that you need to do. For more complex tasks,
you may want to override some of the functions in the base `Processor` class implemented in
`processors/processor.py`. `processors/text_classification.py` gives a good example of a simpler
task, and `processors/named_entity_recognition.py` gives a good example of a more complicated task.

(3) Import this module (text_classification.py) in `__init__.py`
For example, in this file `explainaboard/processors/__init__.py`, we have: 
```python
from . import text_classification
```

## Finally, create a Unittest module for your task

(1) Create a new python file `test_text_classification.py` in the folder `explainaboard/tests/`

(2) Implement a unittest for this task referencing that of other similar tasks

