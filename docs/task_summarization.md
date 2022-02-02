# Analyzing Summarization Tasks

In this file we describe how to analyze models for text summarization, for example those trained and tested on
[CNN/Daily Mail](http://datalab.nlpedia.ai/#/normal_dataset/6176883933e51a7edda9dd68/dataset_metadata).


## Data Preparation

In order to perform analysis of your results, they should be in the following
TSV format:

```
input \t reference_summary \t predicted_summary
```

Let's say we have one system output file: 
* [cnndm_mini.bart](https://github.com/neulab/ExplainaBoard/blob/main/data/system_outputs/cnndm/cnndm_mini.bart) 



## Performing Basic Analysis

In order to perform your basic analysis, we can run the following command:

```shell
    explainaboard --task summarization --system_outputs ./data/system_outputs/cnndm/cnndm_mini.bart > report.json
```
where
* `--task`: denotes the task name. 
* `--system_outputs`: denote the path of system outputs. Multiple one should be 
  separated by comma, for example, system1,system2 (no space)
* `--dataset`: optional, denotes the dataset name
* `report.json`: the generated analysis file with json format. Tips: you can use a json viewer
                  like [this one](http://jsonviewer.stack.hu/) or Python's `python -m json.tool` to convert
                  the JSON into a prettified and readable format.



## Bucketing Features
* `source_len`: the length of the source document
* `compression`: the compression ratio `len(src)/len(ref)`
* `copy_len`: TODO
* `coverage`: TODO
* `novelty`: TODO
