# Analyzing Grammatical Error Correction

Grammatical error correction is the task of correcting different kinds of errors in text, such as spelling errors. If you're interested in how datasets for this task look like, you can
perform the following command after installing [`DataLab`](https://github.com/ExpressAI/DataLab#installation)
```python
from datalabs import load_dataset
dataset = load_dataset("gaokao2018_np1", "writing_grammar")
dataset['test'][0]
```
 
In what follows, we will describe how to analyze grammatical error correction systems. 




## Data Preparation

In order to perform an analysis of your results, your system outputs should be arranged into following
format:

```
[
  {
    "predicted_edits": {
      "start_idx": [8, 17, 39],
      "end_idx": [8, 18, 40],
      "corrections": [
        ["the"],
        ["found"],
        ["other"],
        ]
    }

  }
]
```
where 
* the `len(start_idx) == len(end_idx) == len(corrections)`, representing how many corrections your
your systems think should be made
* the combination of start_idx and end_idx (e.g., (`start_idx[i]`, `end_idx[i]`)) tells the where should be corrected in the original text.
* the value of corrections (`corrections[i]`) tells what correction should be made




## Performing Basic Analysis

Let's say we have one system output file:
* [rst_2018_quanguojuan1_gec.json](https://github.com/neulab/ExplainaBoard/TBC) 

The below example loads the `gaokao2018_np1` dataset (with the subdataset name of `writing-grammar`) from DataLab:
```shell
explainaboard --task grammatical-error-correction --dataset gaokao2018_np1 --sub_dataset writing-grammar --metrics SeqCorrectScore --system_outputs ./explainaboard/tests/artifacts/gaokao/rst_2018_quanguojuan1_gec.json > report.json
```

where
* `--task`: denotes the task name, you can find all supported task names [here](https://github.com/neulab/ExplainaBoard/blob/main/docs/cli_interface.md)
* `--system_outputs`: denote the path of system outputs. Multiple one should be
  separated by space, for example, system1 system2
* `--dataset`: denotes the dataset name
* `--dataset`: denotes the subdataset name
* `--metrics`: represent(s) evaluated metrics being used
* `report.json`: the generated analysis file with json format. Tips: use a json viewer
  like [this one](http://jsonviewer.stack.hu/) for better interpretation.

Alternatively, you can load the dataset from an existing file using the
`--custom_dataset_paths` option
