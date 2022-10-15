# Final Check when You're Contributing as an SDK Developer

Hi, first, thanks for your brilliant contribution to ExplainaBoard SDK :smiley: !

This doc provides some friendly reminders for ExplainaBoard developers so that all
relevant scripts and docs can be kept up-to-date whenever new functionalities have been
implemented in ExplainaBoard.

## When you add a new task

- please register your task in the script [`tasks.py`](https://github.com/neulab/ExplainaBoard/blob/main/explainaboard/tasks.py)
- please add unit tests for your task in the folder [`tests`](https://github.com/neulab/ExplainaBoard/tree/main/integration_tests)
- please update [`cli_interface.md'](https://github.com/neulab/ExplainaBoard/blob/main/docs/cli_interface.md)
  to add the cli information of your task

## When you add a new metric or re-naming evaluation metrics

- please add a unittest in [`test_metric.py`](https://github.com/neulab/ExplainaBoard/blob/main/integration_tests/test_metric.py)
- please update the metric information in the relevant [processors](https://github.com/neulab/ExplainaBoard/blob/main/explainaboard/processors)

## When you add a new supported format

- please add a unittest in the folder [`tests`](https://github.com/neulab/ExplainaBoard/tree/main/integration_tests)
- please update the loader for appropriate tasks in [loaders](https://github.com/neulab/ExplainaBoard/blob/main/explainaboard/loaders)
