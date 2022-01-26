# Add New Tasks

Let's take `text_classification` as an example to show how to add a new task for ExplainaBoard.

### Task and format Declaration
(1) All the supported tasks are listed in **tasks.py**. If your task name is not listed in the file,
please add your task to `TaskType` (enum) and the task list. Task names can not contain `space` and 
different words should be connected using `-`.
(2) if the format of your task's dataset is not covered by `FileType` in the file:
`explainaboard/constants.py`
please manually add the new format.


## Add data loader for your task

(1) creat a new python file `text_classification.py` in the folder `explainaboard/loaders/`
(2) in this file, we need:
* creat a data loader for text classification task inheriting from the class `Loader`
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

(3) import this module (text_classification.py) in `__init__.py`
For example, in this file `explainaboard/loaders/__init__.py`, we have:
```python
from . import text_classification
```


## Creat Builder module for your task
(1) creat a new python file, `text_classification.py` in the folder: `explainaboard/builders/`:
Implement it



## Creat Processor module for your task

(1) creat a new python file `text_classification.py` in the folder `explainaboard/processors/`
(2) in this file, we need:
* creat a processor for text classification task inheriting from the class `Processor`
* define features that you aim to use for this task
* re-implement constructor function
Specifically:
 
(3) import this module (text_classification.py) in `__init__.py`
For example, in this file `explainaboard/processors/__init__.py`, we have: 
```python
from . import text_classification
```