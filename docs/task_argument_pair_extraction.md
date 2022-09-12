# Analyzing the Argument Pair Extraction (APE) Task

In this file we describe how to analyze APE models.
We will give an example using the  [ape](https://github.com/ExpressAI/DataLab/blob/main/datasets/ape/ape.py) dataset, but other datasets
can be analyzed in a similar way.

## Data Preparation

To perform analysis of your results, usually two types of files should be pre-trained, which we will
detailed below.

### Format of `Dataset` File
`Dataset` file usually consists of test samples together with true labels (or references in text generation
tasks). 
In this task, the following specific formats are supported 

* (1) `datalab`
    * if your datasets have been supported by [datalab](https://github.com/ExpressAI/DataLab/tree/main/datasets),
    you fortunately don't need to prepare the dataset. 
    Instead, you just need to remember the dataset name for later use.
    * if your datasets haven't been supported by datalab but you want it supported, you can follow this 
    [doc](https://github.com/ExpressAI/DataLab/blob/main/docs/SDK/add_new_datasets_into_sdk.md) to add them.

    * you can examine the specific format organized in datalab by following commands:
    ```python
    from datalabs import load_dataset
    dataset = load_dataset("ape")
    print(dataset["test"][0])

    ```

### Format of `System Output` File

`System output` file usually only composed of predicted labels (or hypothesis, e.g., system-generated text),
but sometimes `system output` will also contains test samples, such as `CoNLL` format in sequence labeling tasks.
In this task, your system outputs should be as follows:


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


