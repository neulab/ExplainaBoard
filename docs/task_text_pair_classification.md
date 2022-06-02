# Analyzing Text Pair Classification


In this file we describe how to analyze text pair classification models,
such as natural language inference (NLI), paraphrase identification etc.
We will give an example using the `nature-language-inference` 
[SNLI](https://nlp.stanford.edu/projects/snli/)

## Data Preparation

In order to perform analysis of your results, your system outputs should be one
predicted label per line:

```
predicted_label
```

Let's say we have one system output file from a RoBERTa model. 
* [snli.bert](https://github.com/neulab/ExplainaBoard/blob/main/data/system_outputs/snli/snli-roberta-output.txt) 



## Performing Basic Analysis

The below example loads the `snli` dataset from DataLab:
```shell
explainaboard --task text-pair-classification --dataset snli --system_outputs ./data/system_outputs/snli/snli-roberta-output.txt
```

where
* `--task`: denotes the task name, you can find all supported task names [here](https://github.com/neulab/ExplainaBoard/blob/main/docs/cli_interface.md)
* `--system_outputs`: denote the path of system outputs. Multiple one should be
  separated by space, for example, system1 system2
* `--dataset`: denotes the dataset name
* `report.json`: the generated analysis file with json format. Tips: use a json viewer
  like [this one](http://jsonviewer.stack.hu/) for better interpretation.

Alternatively, you can load the dataset from an existing file using the
`--custom_dataset_paths` option

```shell
explainaboard --task text-pair-classification --custom_dataset_paths ./data/system_outputs/snli/snli-dataset.tsv --system_outputs ./data/system_outputs/snli/snli-roberta-output.txt
```

in which case the file format of this file is TSV

```
text1 \t text2 \t true_label
```
