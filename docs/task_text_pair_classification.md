# Analyzing Text Pair Classification

Before diving into the detail of this doc, you're strongly recommended to know [some
important concepts about system analyses](concepts_about_system_analysis.md).


In this file we describe how to analyze text pair classification models,
such as natural language inference (NLI), paraphrase identification etc.
We will give an example using the `nature-language-inference` 
[SNLI](https://nlp.stanford.edu/projects/snli/)

## Data Preparation

### Format of `Dataset` File

* (1) `tsv` (without column names at the first row), see one [example](https://github.com/neulab/ExplainaBoard/blob/main/data/system_outputs/snli/snli-dataset.tsv)
```python
A man playing an electric guitar on stage.   A man playing banjo on the floor.  contradiction
A man playing an electric guitar on stage.   A man is performing for cash.  neutral
...
```
* (2) `json` (basically, it's a list of dictionaries with three keys: `text1`, `text2` and `true_label`)
```json
[
  {"text1": "A man playing an electric guitar on stage.", "text2": "A man playing banjo on the floor.", "true_label": "contradiction"},
  {"text1": "A man playing an electric guitar on stage.", "text2": "A man is performing for cash.", "true_label": "neutral"},
  ...
]
```

* (3) `datalab`: if your datasets have been supported by [datalab](https://github.com/ExpressAI/DataLab/tree/main/datasets),
    you fortunately don't need to prepare the dataset. 

### Format of `System Output` File

In this task, your system outputs should be one predicted label per line:

```
predicted_label
```

Let's say we have one system output file from a RoBERTa model. 
* [snli.bert](https://github.com/neulab/ExplainaBoard/blob/main/data/system_outputs/snli/snli-roberta-output.txt) 



## Performing Basic Analysis

The below example loads the `snli` dataset from DataLab:
```shell
explainaboard --task text-pair-classification --dataset snli --system-outputs ./data/system_outputs/snli/snli-roberta-output.txt
```

where
* `--task`: denotes the task name, you can find all supported task names [here](https://github.com/neulab/ExplainaBoard/blob/main/docs/cli_interface.md)
* `--system-outputs`: denote the path of system outputs. Multiple one should be
  separated by space, for example, system1 system2
* `--dataset`: denotes the dataset name
* `report.json`: the generated analysis file with json format. Tips: use a json viewer
  like [this one](http://jsonviewer.stack.hu/) for better interpretation.

Alternatively, you can load the dataset from an existing file using the
`--custom-dataset-paths` option

```shell
explainaboard --task text-pair-classification --custom-dataset-paths ./data/system_outputs/snli/snli-dataset.tsv --system-outputs ./data/system_outputs/snli/snli-roberta-output.txt
```

in which case the file format of the custom dataset file (`snli-dataset.tsv`) is TSV

```
text1 \t text2 \t true_label
```
