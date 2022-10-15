# Analyzing the Argument Pair Identification (API) Task

In this file we describe how to analyze API models.
We will give an example using the [ape](https://github.com/ExpressAI/DataLab/blob/main/datasets/iapi/iapi.py)
dataset, but other datasets can be analyzed in a similar way.

## Data Preparation

In order to perform analysis of your results, they should be in the following text
format:

```text
0
1
2
0
1
0
0
```

where the number represents the index of the predicted reply.

An example system output file is here: [predictions.txt](../../data/system_outputs/iapi/predictions.txt)

## Performing Basic Analysis

In order to perform your basic analysis, we can run the following command:

```shell
explainaboard --task argument-pair-identification --dataset iapi --system-outputs data/system_outputs/iapi/predictions.txt > report.json

```

where

* `--task`: denotes the task name, you can find all supported task names
  [here](https://github.com/neulab/ExplainaBoard/blob/main/docs/supported_tasks.md)
* `--system-outputs`: denote the path of system outputs. Multiple paths should be
  separated by a space, for example, system1 system2
* `--dataset`: denotes the dataset name
* `report.json`: the generated analysis file with json format. You can find the file
  [here](https://github.com/ExpressAI/ExplainaBoard/blob/main/data/reports/report.json).
  Tips: use a json viewer like [this one](http://jsonviewer.stack.hu/) for better
  interpretation.
