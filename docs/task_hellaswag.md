# Analyzing Text Pair Classification


In this file we describe how to analyze models trained on multi-choice QA dataset:
[`hellaswag`](https://huggingface.co/datasets/hellaswag).


## Data Preparation

In order to perform analysis of your results, they should be in the following
tsv format (don't need to contain the column names that the first line):

```
id \t predicted_label
```
where 
* `id`: represents the sample id in huggingface validation dataset.
* `predicted_label`: denotes the predicted answer index (0,1,2,3,)

Let's say we have one system output file: 
* [hellaswag.random](https://github.com/ExpressAI/ExplainaBoard/blob/main/data/system_outputs/hellaswag/hellaswag.random) 



## Performing Basic Analysis

In order to perform your basic analysis, we can run the following command:

```shell
    explainaboard --task hellaswag --system_outputs ../data/system_outputs/hellaswag/hellaswag.random > report.json
```
where
* `--task`: denotes the task name. this could be applied for any sentence pair classification subtasks.
* `--system_outputs`: denote the path of system outputs. Multiple one should be 
  separated by comma, for example, system1,system2 (no space)
* `--dataset`:optional, denotes the dataset name
* `report.json`: the generated analysis file with json format. . Tips: use a json viewer
                  like [this one](http://jsonviewer.stack.hu/) for better interpretation.



## Bucketing Features
* `activity_label`: [The ActivityNet or WikiHow label for this example](https://github.com/rowanz/hellaswag/tree/master/data)
* `ind`: dataset ID
* `ctx_length`: the length of the full context `ctx`
* `ctx_a_length_divided_b`: ctx_a_length/ctx_b_length
* `true_answer_length`: the length of true answer
* `similarity_ctx_true_answer`: the semantic similarity between full context `ctx` and the true answer