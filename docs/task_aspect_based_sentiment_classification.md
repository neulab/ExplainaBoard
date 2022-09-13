# Analyzing Aspect-based Sentiment Classification

Before diving into the detail of this doc, you're strongly recommended to know [some
important concepts about system analyses](concepts_about_system_analysis.md).


In this file we describe how to analyze aspect-based sentiment classification models.
We will give an example using the `aspect-based-sentiment-classification` [laptop](https://github.com/neulab/ExplainaBoard/blob/main/data/system_outputs/absa/test-aspect.tsv) dataset, but other datasets
can be analyzed in a similar way.

## Data Preparation

 

### Format of `Dataset` File


* (1) `tsv` (without column names at the first row), see one [example](https://github.com/neulab/ExplainaBoard/blob/main/data/system_outputs/absa/absa-dataset.tsv)
```python
Boot time	 Boot time  is super fast, around anywhere from 35 seconds to 1 minute.	positive
Windows 8	Did not enjoy the new  Windows 8  and  touchscreen functions .	negative
...
```
where the first 1st, 2nd, 3rd column represent aspect text, sentence and true label respectively.


* (2) `json` (basically, it's a list of dictionaries with three keys: `aspect`, `text` and `true_label`)
```json
[
  {"aspect":"Boot time", "text": "Boot time  is super fast, around anywhere from 35 seconds to 1 minute.", "true_label": "positive"},
  ...
]
```

* (3) `datalab`: if your datasets have been supported by [datalab](https://github.com/ExpressAI/DataLab/tree/main/datasets),
    you fortunately don't need to prepare the dataset. 


### Format of `System Output` File
In this task, your system outputs should be as follows:


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
explainaboard --task aspect-based-sentiment-classification --custom-dataset-paths ./data/system_outputs/absa/absa-dataset.txt --system-outputs ./data/system_outputs/absa/absa-example-output.tsv > report.json
```
where
* `--task`: denotes the task name, you can find all supported task names [here](https://github.com/neulab/ExplainaBoard/blob/main/docs/supported_tasks.md)
* `--system-outputs`: denote the path of system outputs. Multiple one should be 
  separated by space, for example, system1 system2
* `report.json`: the generated analysis file with json format. You can find the file [here](https://github.com/neulab/ExplainaBoard/blob/main/data/reports/report_absa.json). Tips: use a json viewer
                  like [this one](http://jsonviewer.stack.hu/) for better interpretation.

