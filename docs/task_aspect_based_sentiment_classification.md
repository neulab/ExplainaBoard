# Analyzing Aspect-based Sentiment Classification

In this file we describe how to analyze aspect-based sentiment classification models.
We will give an example using the `aspect-based-sentiment-classification` [laptop](https://github.com/neulab/ExplainaBoard/blob/main/data/system_outputs/absa/test-aspect.tsv) dataset, but other datasets
can be analyzed in a similar way.

## Data Preparation

In order to perform analysis of your results, they should be in the following
text format with one predicted label per line.

```
predicted_label
```

Let's say we have several files such as 
* [absa-example-output.tsv](https://github.com/neulab/ExplainaBoard/blob/main/data/system_outputs/absa/absa-example-output.tsv) 
 

etc. from different systems.


## Performing Basic Analysis

If your dataset exists in DataLab you can read it directly from there. However, here
we will give an example of using a custom dataset, which takes this form:
```
aspect \t sentence \t true_label 
```

In order to perform your basic analysis, we can run the following command:

```shell
explainaboard --task aspect-based-sentiment-classification --custom_dataset_paths ./data/system_outputs/absa/absa-dataset.txt --system_outputs ./data/system_outputs/absa/absa-example-output.tsv > report.json
```
where
* `--task`: denotes the task name, you can find all supported task names [here](https://github.com/neulab/ExplainaBoard/blob/main/docs/supported_tasks.md)
* `--system_outputs`: denote the path of system outputs. Multiple one should be 
  separated by space, for example, system1 system2
* `report.json`: the generated analysis file with json format. You can find the file [here](https://github.com/neulab/ExplainaBoard/blob/main/data/reports/report_absa.json). Tips: use a json viewer
                  like [this one](http://jsonviewer.stack.hu/) for better interpretation.

