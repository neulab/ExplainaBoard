# Analyzing Aspect-based Sentiment Classification

In this file we describe how to analyze aspect-based sentiment classification models.
We will give an example using the `aspect-based-sentiment-classification` [laptop](https://github.com/neulab/ExplainaBoard/blob/main/data/system_outputs/absa/test-aspect.tsv) dataset, but other datasets
can be analyzed in a similar way.

## Data Preparation

In order to perform analysis of your results, they should be in the following
tsv format (don't need to contain the column names that the first line):

```
aspect \t sentence \t true_label \t predicted_label
```

Let's say we have several files such as 
* [test-aspect.tsv](https://github.com/neulab/ExplainaBoard/blob/main/data/system_outputs/absa/test-aspect.tsv) 
 

etc. from different systems.


## Performing Basic Analysis

In order to perform your basic analysis, we can run the following command:

```shell
    explainaboard --task aspect-based-sentiment-classification --system_outputs ./data/system_outputs/absa/test-aspect.tsv > report.json
```
where
* `--task`: denotes the task name, you can find all supported task names [here](https://github.com/ExpressAI/ExplainaBoard/blob/feat_docs_task/docs/existing_supports.md)
* `--system_outputs`: denote the path of system outputs. Multiple one should be 
  separated by space, for example, system1 system2
* `report.json`: the generated analysis file with json format. You can find the file [here](https://github.com/ExpressAI/ExplainaBoard/blob/main/data/reports/report_absa.json). Tips: use a json viewer
                  like [this one](http://jsonviewer.stack.hu/) for better interpretation.




Now let's look at the results to see what sort of interesting insights we can
glean from them.

TODO: add insights

 
