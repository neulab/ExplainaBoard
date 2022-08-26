# Analyzing Text Classification

In this file we describe how to analyze text classification models.
We will give an example using the `text-classification` [sst2](https://github.com/ExpressAI/ExplainaBoard/tree/main/data/datasets/sst2) dataset, but other datasets
can be analyzed in a similar way.

## Data Preparation

In order to perform analysis of your results, your system outputs should be one
predicted label per line:

```
predicted_label
```

Let's say we have several files such as 
* [sst2-lstm.tsv](https://github.com/neulab/ExplainaBoard/blob/main/data/system_outputs/sst2/sst2-lstm-output.txt) 
* [sst2-cnn.tsv](https://github.com/neulab/ExplainaBoard/blob/main/data/system_outputs/sst2/sst2-cnn-output.txt)

etc. from different systems.


## Performing Basic Analysis

The below example loads the `sst2` dataset from DataLab:
```shell
explainaboard --task text-classification --dataset sst2 --system-outputs ./data/system_outputs/sst2/sst2-lstm-output.txt
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
explainaboard --task text-classification --custom-dataset-paths ./data/system_outputs/sst2/sst2-dataset.tsv --system-outputs ./data/system_outputs/sst2/sst2-lstm-output.txt
```

in which case the file format of this file is TSV
```
text \t true_label
```

## Advanced Analysis Options

One also can perform pair-wise analysis:
```shell
explainaboard --task text-classification --dataset sst2 --system-outputs ./data/system_outputs/sst2/sst2-lstm-output.txt ./data/system_outputs/sst2/sst2-cnn-output.txt > report.json
```
where two system outputs are fed separated by space.
* `report.json`: the generated analysis file with json format, whose schema is similar 
  to the above one with single system evaluation except that
  all performance values are obtained using the sys1 subtract sys2.
