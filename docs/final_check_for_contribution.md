# Final Check when You're Contributing SDK


Hi, first, thanks for your brilliant contribution to ExplainaBoard SDK :smiley: !

This doc provides some friendly reminder for ExplainaBoard developers so that all relevant scripts and docs
could be kept updated timely whenever new functionalities have been implemented in ExplainaBoard.



- #### When you add a new task:
    - please make your task registered in the script [`tasks.py`](https://github.com/neulab/ExplainaBoard/blob/main/explainaboard/tasks.py)
    - please add a unittest for your task in the folder [`tests`](https://github.com/neulab/ExplainaBoard/tree/main/explainaboard/tests)
    - please update [`cli_interface.md'](https://github.com/neulab/ExplainaBoard/blob/main/docs/cli_interface.md) to add the cli information of your task
- #### When you add a new metric or re-naming evaluation metrics,
    - please update the `supported_metrics` information in [`tasks.py`](https://github.com/neulab/ExplainaBoard/blob/main/explainaboard/tasks.py)
    - please add a unittest in [`test_metric.py`](https://github.com/neulab/ExplainaBoard/blob/main/explainaboard/tests/test_metric.py)
- #### When you add a new supported format,
    - please modify the `supported_tasks` information [`tasks.py`](https://github.com/neulab/ExplainaBoard/blob/main/explainaboard/tasks.py)
    - please add a unittest in the folder [`tests`](https://github.com/neulab/ExplainaBoard/tree/main/explainaboard/tests)