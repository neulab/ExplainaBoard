# Analyzing the Argument Pair Extraction (APE) Task

Before diving into the detail of this doc, you're strongly recommended to know [some
important concepts about system analyses](concepts_about_system_analysis.md).

In this file we describe how to analyze APE models.
We will give an example using the  [ape](https://github.com/ExpressAI/DataLab/blob/main/datasets/ape/ape.py) dataset, but other datasets
can be analyzed in a similar way.

## Data Preparation

 

### Format of `Dataset` File
 

* (1) `datalab`: if your datasets have been supported by [datalab](https://github.com/ExpressAI/DataLab/tree/main/datasets),
    you fortunately don't need to prepare the dataset. For example, you can examine the specific format organized in datalab by following commands:
    ```python
    from datalabs import load_dataset
    dataset = load_dataset("ape")
    print(dataset["test"][0])

    ```

### Format of `System Output` File

In order to perform analysis of your results, they should be in the following conll format:

```
O	O
O	O
O	O
O	O
O	O
O	O
O	O
O	O
B-1	B-1
I-1	I-1
B-2	I-1
I-2	I-1
I-2	I-1
I-2	I-1
I-2	I-1
I-2	I-1
```
where the first column represents true tag, the second column represents predicted tag.



An example system output file is here: [ape_predictions.txt](../../data/system_outputs/ape/ape_predictions.txt)

  

## Performing Basic Analysis

In order to perform your basic analysis, we can run the following command:

```shell
explainaboard --task argument-pair-extraction --dataset ape --system-outputs ./data/system_outputs/ape/ape_predictions.txt
```
where
* `--task`: denotes the task name, you can find all supported task names [here](https://github.com/neulab/ExplainaBoard/blob/main/docs/supported_tasks.md)
* `--system-outputs`: denote the path of system outputs. Multiple one should be 
  separated by space, for example, system1 system2
* `--dataset`: denotes the dataset name
* `report.json`: the generated analysis file with json format. You can find the file [here](https://github.com/ExpressAI/ExplainaBoard/blob/main/data/reports/report.json). Tips: use a json viewer
                  like [this one](http://jsonviewer.stack.hu/) for better interpretation.


