# Analyzing Text Pair Classification


In this file we describe how to analyze text pair classification models,
such as natural language inference (NLI), paraphrase identification etc.
We will give an example using the `nature-language-inference` 
[SNLI](https://nlp.stanford.edu/projects/snli/)

## Data Preparation

In order to perform analysis of your results, they should be in the following
tsv format (don't need to contain the column names that the first line):

```
text1 \t text2 \t true_label \t predicted_label
```

Let's say we have oen system output file from BERT model. 
* [snli.bert](https://github.com/ExpressAI/ExplainaBoard/blob/main/data/system_outputs/snli/snli.bert) 



## Performing Basic Analysis

In order to perform your basic analysis, we can run the following command:

```shell
    explainaboard --task text-pair-classification --system_outputs ./data/system_outputs/snli/snli.bert > report.json
```
where
* `--task`: denotes the task name. this could be applied for any sentence pair classification subtasks.
* `--system_outputs`: denote the path of system outputs. Multiple one should be 
  separated by space, for example, system1 system2
* `--dataset`:optional, denotes the dataset name
* `report.json`: the generated analysis file with json format. . Tips: use a json viewer
                  like [this one](http://jsonviewer.stack.hu/) for better interpretation.



## Bucketing Features
* `label`: the relationship types of two texts
* `text1_length`: the length of `text1`
* `text2_length`: the length of `text2`
* `similarity`: the semantic similarity of two texts measured by `bleu`
* `text1_divided_text2`: text1_length/text2_length