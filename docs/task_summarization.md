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
explainaboard --task summarization --system_outputs ./data/system_outputs/cnndm/cnndm_mini.bart --metrics bart_score_summ rouge2
```

where
* `--task`: denotes the task name. 
* `--system_outputs`: denote the path of system outputs. Multiple one should be 
  separated by space, for example, system1 system2
* `--dataset`: optional, denotes the dataset name
* `--metrics`: optional, different metrics should be separated by space. See [more supported metrics](https://github.com/neulab/ExplainaBoard/blob/main/docs/supported_tasks.md#summarization)
* `report.json`: the generated analysis file with json format. Tips: you can use a json viewer
                  like [this one](http://jsonviewer.stack.hu/) or Python's `python -m json.tool` to convert
                  the JSON into a prettified and readable format.



## Bucketing Features
* `source_len`: the length of the source document
* `compression`: the compression ratio `len(src)/len(ref)`
* [`copy_len`](https://aclanthology.org/2020.findings-emnlp.329.pdf): measures the average length of segments in summary copied from source document.
* [`coverage`](https://aclanthology.org/2020.findings-emnlp.329.pdf): illustrates the overlap rate between document and summary, it is defined as the proportion of the copied segments in
summary.
* [`novelty`]((https://aclanthology.org/2020.findings-emnlp.329.pdf)): is defined as the proportion of segments in the summaries that haven’t
appeared in source documents. The segments is instantiated as 2-grams.
