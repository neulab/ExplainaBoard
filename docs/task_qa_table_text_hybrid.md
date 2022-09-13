# Analyzing QA over Hybrid Tabular/Textual Content

Before diving into the detail of this doc, you're strongly recommended to know [some
important concepts about system analyses](concepts_about_system_analysis.md).



In this file we describe how to analyze QA models trained on datasets 
with a hybrid of tabular and textual context.


## Data Preparation

 
### Format of `Dataset` File
In this task, only the `datalab` format is supported so far:

* (1) `datalab`: if your datasets have been supported by [datalab](https://github.com/ExpressAI/DataLab/tree/main/datasets),
    you fortunately don't need to prepare the dataset. 

### Format of `System Output` File

In this task, your system outputs should be:

```json
[
{"q_id": "b64f475c0e1cc1e653d0b239f09da0d7", "answer": ["48.4"], "scale": "million"},
{"q_id": "a7457ad860d3137ebc05538509ad8ac8", "answer": 31.95, "scale": "million"},
{"q_id": "b484c510cb9ee8dd1f9524e0fad578dd", "answer": 35.15, "scale": "million"},
...
]
```
where each line represents one predicted answer. Specifically,
* `q_id` represents the question id
* `answer` denotes the predicted answer
* `scale` is the predicted scale.

Check this [page](https://www.datafountain.cn/competitions/573/datasets) to know detailed 
definition of `scale`.

An example system output file is here: [predictions_list.json](https://explainaboard.s3.amazonaws.com/system_outputs/qa_table_text_hybrid/predictions_list.json)

 
## Performing Basic Analysis

In order to perform your basic analysis, we can run the following command:

```shell
explainaboard --task qa-tat --output-file-type json --dataset tat_qa --system-outputs predictions_list.json > report.json
```
where
* `--task`: denotes the task name, you can find all supported task names [here](https://github.com/neulab/ExplainaBoard/blob/main/docs/supported_tasks.md)
* `--system-outputs`: denote the path of system outputs. Multiple one should be 
  separated by space, for example, system1 system2
* `--dataset`:denotes the dataset name
* `report.json`: the generated analysis file with json format. You can find the file [here](https://github.com/ExpressAI/ExplainaBoard/blob/main/data/reports/report.json). Tips: use a json viewer
                  like [this one](http://jsonviewer.stack.hu/) for better interpretation.


