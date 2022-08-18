# Analyzing Open-domain QA

In this file we describe how to analyze open-domain QA models.
We will give an example using the  [natural_questions_comp_gen](https://github.com/ExpressAI/DataLab/blob/main/datasets/natural_questions_comp_gen/natural_questions_comp_gen.py) dataset, but other datasets
can be analyzed in a similar way.

## Data Preparation

In order to perform analysis of your results, they should be in the following json format:

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
explainaboard --task qa-open-domain --dataset natural_questions_comp_gen   --system_outputs ./data/system_outputs/qa_open_domain/test.dpr.nq.txt  > report.json
```
where
* `--task`: denotes the task name, you can find all supported task names [here](https://github.com/neulab/ExplainaBoard/blob/main/docs/supported_tasks.md)
* `--system_outputs`: denote the path of system outputs. Multiple one should be 
  separated by space, for example, system1 system2
* `--dataset`:denotes the dataset name
* `report.json`: the generated analysis file with json format. You can find the file [here](https://github.com/ExpressAI/ExplainaBoard/blob/main/data/reports/report.json). Tips: use a json viewer
                  like [this one](http://jsonviewer.stack.hu/) for better interpretation.

