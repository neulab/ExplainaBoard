# Analyzing Knowledge Graph Link Tail Prediction Task


In this file we describe how to analyze models trained to predict the tail entity on knowledge graph link prediction tasks, for example
[`fb15k-237`](https://www.microsoft.com/en-us/download/details.aspx?id=52312).


## Data Preparation

In order to perform analysis of your results, they should be in the following
JSON format:

```json
{
    "<head 1>\t<relation 1>\t<tail 1>": [<rank-1 tail prediction>, ... , <rank-5 tail prediction>],
    "<head 2>\t<relation 2>\t<tail 2>": [<rank-1 tail prediction>, ... , <rank-5 tail prediction>],
    
}
```
where each record is indexed by a ground-truth link triple `(<head>, <relation>, <tail>)` in the knowledge graph described as tab-separated text; and the top-5 model-predicted tail entities as values.

Let's say we have one system output file: 
* [test-kg-link-tail-prediction.json](https://github.com/neulab/ExplainaBoard/blob/main/data/system_outputs/fb15k-237/test-kg-link-tail-prediction.json) 



## Performing Basic Analysis

In order to perform your basic analysis, we can run the following command:

```shell
    explainaboard --task kg-link-tail-prediction --system_outputs ./data/system_outputs/fb15k-237/test-kg-link-tail-prediction.json --dataset fb15k_237 > report.json

or

    explainaboard --task kg-link-tail-prediction --system_outputs ./data/system_outputs/fb15k-237/test-kg-link-tail-prediction.json > report.json
```
where
* `--task`: denotes the task name. 
* `--system_outputs`: denote the path of system outputs. Multiple one should be 
  separated by space, for example, system1 system2
* `--dataset`:optional, denotes the dataset name
* `report.json`: the generated analysis file with json format. Tips: use a json viewer like [`this one`](http://jsonviewer.stack.hu/) for better interpretation.



## Bucketing Features
* Toy feature `tail_entity_length`: the number of words in `true_tail`
* More meaningful features to be added soon

## User-Defined Features
To provide your own bucketing features, submit a system output containing a declaration of your user-defined features (their names, data types, and number of buckets), along with your predictions on test examples. Make sure each test example contains a key for each feature defined in your configuration. Refer to the following example:

```json
{
    "user_defined_features_configs":
    {
        "user_defined_feature_1": {
            "dtype": "string",
            "description": "just a toy string feature for testing",
            "num_buckets": 8
        },
        "user_defined_feature_2": {
            "dtype": "float",
            "description": "just a toy float feature for testing",
            "num_buckets": 4
        },
        ...
        "user_defined_feature_n": {
            "dtype": "float",
            "description": "just a toy float feature for testing",
            "num_buckets": 6
        },
    },


    "predictions":
    {
        "<head 1>\t<relation 1>\t<tail 1>": {
            "predictions": [<rank-1 tail prediction>, ... , <rank-5 tail prediction>],
            "user_defined_feature_1": "string_value_1",
            "user_defined_feature_2": 0.8374,
            ...
            "user_defined_feature_n": 0.5984
        },
        "<head 2>\t<relation 2>\t<tail 2>": {
            "predictions": [<rank-1 tail prediction>, ... , <rank-5 tail prediction>],
            "user_defined_feature_1": "string_value_2",
            "user_defined_feature_2": 0.1422,
            ...
            "user_defined_feature_n": 0.3795
        },
        ...
    }
}
```

An example system output is provided, and you can test it using the following command:

```shell
    explainaboard --task kg-link-tail-prediction --system_outputs ./data/system_outputs/fb15k-237/test-user-defined-features-prediction-short.json --dataset fb15k_237 > report.json

or

    explainaboard --task kg-link-tail-prediction --system_outputs ./data/system_outputs/fb15k-237/test-user-defined-features-prediction-short.json > report.json
```

