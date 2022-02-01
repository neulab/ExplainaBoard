# Analyzing Extractive QA Task


In this file we describe how to analyze models trained on extractive QA datasets, for example
[`squad`](http://datalab.nlpedia.ai/#/normal_dataset/6163a29beb9872f33252b01b/dataset_samples).


## Data Preparation

In order to perform analysis of your results, they should be in the following
JSON format:

```json
{
    "56beb4343aeaaa14008c925b": "308",
    "56beb4343aeaaa14008c925c": "136",
    "56beb4343aeaaa14008c925d": "(118)",
    
}
```
where 
* `56beb4343aeaaa14008c925b`: represents one question id.
* `308`: denotes the corresponding predicted answer.

Let's say we have one system output file: 
* [test-qa-squad.json](https://github.com/neulab/ExplainaBoard/blob/main/data/system_outputs/squad/test-qa-squad.json) 



## Performing Basic Analysis

In order to perform your basic analysis, we can run the following command:

```shell
    explainaboard --task extractive-qa --system_outputs ./data/system_outputs/squad/test-qa-squad.json > report.json
```
where
* `--task`: denotes the task name. 
* `--system_outputs`: denote the path of system outputs. Multiple one should be 
  separated by comma, for example, system1,system2 (no space)
* `--dataset`:optional, denotes the dataset name
* `report.json`: the generated analysis file with json format. . Tips: use a json viewer
                  like [this one](http://jsonviewer.stack.hu/) for better interpretation.



## Bucketing Features
* `context_length`: the length of  `context`
* `question_length`: the length of  `question`
* `answer_length`: the length of  `answer`