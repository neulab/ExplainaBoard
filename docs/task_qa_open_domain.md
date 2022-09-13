# Analyzing Open-domain QA

Before diving into the detail of this doc, you're strongly recommended to know [some
important concepts about system analyses](concepts_about_system_analysis.md).

In this file we describe how to analyze open-domain QA models.
We will give an example using the  [natural_questions_comp_gen](https://github.com/ExpressAI/DataLab/blob/main/datasets/natural_questions_comp_gen/natural_questions_comp_gen.py) dataset, but other datasets
can be analyzed in a similar way.

## Data Preparation

 

### Format of `Dataset` File

* (1) `json` (basically, it's a list of dictionaries with two keys: `question` and `answers`)
```json
[
  {'question': 'who got the first nobel prize in physics', 'answers': ['Wilhelm Conrad Röntgen']},
  {'question': 'when is the next deadpool movie being released', 'answers': ['May 18 , 2018']},
  ...
]
```

* (2) `datalab`: if your datasets have been supported by [datalab](https://github.com/ExpressAI/DataLab/tree/main/datasets),
    you fortunately don't need to prepare the dataset. 

### Format of `System Output` File

In this task, your system outputs should be as follows:

```text
william henry bragg
may 18, 2018
...
```
where each line represents one predicted answer.
An example system output file is here: [test.dpr.nq.txt](https://github.com/neulab/ExplainaBoard/blob/add_customized_features_from_config/data/system_outputs/qa_open_domain/test.dpr.nq.txt)

 

Let's say we have several files such as 
* [gpt2.json](https://github.com/neulab/ExplainaBoard/blob/main/data/system_outputs/fig_qa/gpt2.json) 


etc. from different systems.


## Performing Basic Analysis

In order to perform your basic analysis, we can run the following command:

```shell
explainaboard --task qa-open-domain --dataset natural_questions_comp_gen   --system-outputs ./data/system_outputs/qa_open_domain/test.dpr.nq.txt  > report.json
```
where
* `--task`: denotes the task name, you can find all supported task names [here](https://github.com/neulab/ExplainaBoard/blob/main/docs/supported_tasks.md)
* `--system-outputs`: denote the path of system outputs. Multiple one should be 
  separated by space, for example, system1 system2
* `--dataset`:denotes the dataset name
* `report.json`: the generated analysis file with json format. You can find the file [here](https://github.com/ExpressAI/ExplainaBoard/blob/main/data/reports/report.json). Tips: use a json viewer
                  like [this one](http://jsonviewer.stack.hu/) for better interpretation.


