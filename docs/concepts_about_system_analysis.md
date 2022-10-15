# Concepts about System Analysis

In these docs, we will explain some important concepts that
you need to know before performing system analyses using Explainaboard.

## What are a "dataset" and a "system output"?

To perform an analysis of your results, usually, two types of files should be prepared:
"dataset" and "system output."

### `Dataset`

A "dataset" usually consists of test inputs together with true outputs (e.g.
gold-standard labels for classification tasks or reference outputs for text generation
tasks). For example, in text classification, a `dataset` organized in tsv format may
look like this:

```text
I love this movie   positive
The movie is too long   negative
...
```

### `System output`

`System output` is frequently composed of predicted labels (or hypotheses, e.g.,
system-generated text), but sometimes `system output` will also contain test samples,
such as `CoNLL` format in sequence labeling tasks. For example, in text classification,
the `system output` could be organized in a text format:

```text
positive
negative
...
```

## "Datalab datasets" and "custom datasets"

ExplainaBoard currently supports two sources for datasets: one is through
[Datalab](https://aclanthology.org/2022.acl-demo.18.pdf) (Xiao et al 2022) and the other
is custom.

* if your datasets have been supported by [datalab](https://github.com/ExpressAI/DataLab/tree/main/datasets),
we recommend that you load it through DataLab. This is for several reasons:
 (1) you don't have to prepare the data yourself,
 (2) you can be sure that your accuracy numbers will be comparable with those calculated
 by other people using ExplainaBoard, and
 (3) DataLab datasets support analysis with features calculated over the training set,
  which can be informative.

For example, the following command can be used to perform system analysis on `sst2`
dataset (supported by datalab)

```shell script
explainaboard --task text-classification --dataset sst2 --system-outputs ./data/system_outputs/sst2/sst2-lstm-output.txt
```

If the dataset hasn't been supported by datalab, we then need to prepare the dataset in
a format supported by ExplainaBoard (this varies from task-to-task, see the
task-specific documentation) and specify a file path:

```shell script
explainaboard --task text-classification --custom-dataset-paths ./data/system_outputs/sst2/sst2-dataset.tsv --system-outputs ./data/system_outputs/sst2/sst2-lstm-output.txt
```

* if your datasets haven't been supported by datalab but you want them supported, you
  can follow this [doc](https://github.com/ExpressAI/DataLab/blob/main/docs/SDK/add_new_datasets_into_sdk.md)
  to add them.

## How different formats (e.g. json, tsv, etc.) are supported

In ExplainaBoard, users are allowed to prepare custom datasets with different formats,
such as `json`, `tsv` etc. We will detail this in each task's description.
