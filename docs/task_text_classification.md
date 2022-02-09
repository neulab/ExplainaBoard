# Analyzing Text Classification

In this file we describe how to analyze text classification models.
We will give an example using the `text-classification` [sst2](https://github.com/ExpressAI/ExplainaBoard/tree/main/data/datasets/sst2) dataset, but other datasets
can be analyzed in a similar way.

## Data Preparation

In order to perform analysis of your results, they should be in the following
tsv format (don't need to contain the column names that the first line):

```
text \t true_label \t predicted_label
```

Let's say we have several files such as 
* [sst2-lstm.tsv](https://github.com/ExpressAI/ExplainaBoard/blob/main/data/system_outputs/sst2/sst2-lstm.tsv) 
* [sst2-cnn.tsv](https://github.com/ExpressAI/ExplainaBoard/blob/main/data/system_outputs/sst2/sst2-cnn.tsv)

etc. from different systems.


## Performing Basic Analysis

In order to perform your basic analysis, we can run the following command:

```shell
    explainaboard --task text-classification --system_outputs ./data/system_outputs/sst2/sst2-lstm.tsv > report.json
```
where
* `--task`: denotes the task name, you can find all supported task names [here](https://github.com/ExpressAI/ExplainaBoard/blob/feat_docs_task/docs/existing_supports.md)
* `--system_outputs`: denote the path of system outputs. Multiple one should be 
  separated by space, for example, system1 system2
* `--dataset`:optional, denotes the dataset name
* `report.json`: the generated analysis file with json format. You can find the file [here](https://github.com/ExpressAI/ExplainaBoard/blob/main/data/reports/report.json). Tips: use a json viewer
                  like [this one](http://jsonviewer.stack.hu/) for better interpretation.




Now let's look at the results to see what sort of interesting insights we can
glean from them.

TODO: add insights

## Advanced Analysis Options

One also can perform pair-wise analysis:
```shell
explainaboard --task text-classification --system_outputs ./data/system_outputs/sst2/sst2-lstm.tsv ./data/system_outputs/sst2/sst2-cnn.tsv > report.json
```
where two system outputs are fed separated by space.
* `report.json`: the generated analysis file with json format, whose schema is similar to the above one with single system evaluation except that
   all performance values are obtained using the sys1 subtract sys2.