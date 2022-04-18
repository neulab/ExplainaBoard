# Analyzing Knowledge Graph Link Tail Prediction Task


In this file we describe how to analyze models trained to predict the tail entity on knowledge graph link prediction tasks, for example
[`fb15k-237`](https://www.microsoft.com/en-us/download/details.aspx?id=52312).

## Outline
* Evaluation with Build-in Features
    * Data Preparation
    * Perform Analysis with CLI
    * Visualization Locally
* Evaluation with Customized Features
    * Data Preparation
    * Perform Analysis with CLI





## Evaluation with Build-in Features



### Data Preparation
In order to perform analysis of your results, they should be in the following
JSON format:

```json
{
    "1": {
        "gold_head": "/m/08966",
        "gold_predicate": "/travel/travel_destination/climate./travel/travel_destination_monthly_climate/month",
        "gold_tail": "/m/05lf_",
        "predict": "tail",
        "predictions": [
            "/m/05lf_",
            "/m/02x_y",
            "/m/01nv4h",
            "/m/02l6h",
            "/m/0kz1h"
        ]
    },
    "2": {
        "gold_head": "/m/01hww_",
        "gold_predicate": "/music/performance_role/regular_performances./music/group_membership/group",
        "gold_tail": "/m/01q99h",
        "predict": "tail",
        "predictions": [
            "/m/05563d",
            "/m/02vnpv",
            "/m/02r1tx7",
            "/m/017lb_",
            "/m/03c3yf"
        ]
    },
    ...
    
}
```
where
* `gold_head`: true head entity
* `gold_predicate`: true relation
* `gold_tail`: true tail entity
* `predict`: it suggest what type of information (e.g., `head`, `predicate`, `tail`) will be predicted
* `predictions`: a list of predictions

Let's say we have one system output file: 
* [test-kg-prediction-no-user-defined-new.json](https://github.com/neulab/ExplainaBoard/blob/main/explainaboard/tests/artifacts/test-kg-prediction-no-user-defined-new.json) 



### Perform Analysis with CLI

In order to perform your basic analysis, we can run the following command:

```shell
    explainaboard --task kg-link-tail-prediction --system_outputs ./data/system_outputs/fb15k-237/test-kg-prediction-no-user-defined-new.json --dataset fb15k_237 > report.json

or

    explainaboard --task kg-link-tail-prediction --system_outputs ./data/system_outputs/fb15k-237/test-kg-prediction-no-user-defined-new.json > report.json
```
where
* `--task`: denotes the task name. 
* `--system_outputs`: denote the path of system outputs. Multiple one should be 
  separated by space, for example, system1 system2
* `--dataset`:optional, denotes the dataset name
* `report.json`: the generated analysis file with json format. Tips: use a json viewer like [`this one`](http://jsonviewer.stack.hu/) for better interpretation.



### Bucketing Features
* Toy feature `tail_entity_length`: the number of words in `true_tail`
* More meaningful features to be added soon


### Visualization Locally
Once the above command has been successfully conducted, histogram figures will be generated automatically in the folder
`./output/figures/test-kg-prediction-no-user-defined-new/`, where each figure represent a fine-grained evaluation
results along one features (e.g., relation type).
 
We have carefully designed and beautified these figures which 
could be directly applied for paper writing as needed.

One example is shown below,


<img src="./figures/entity_type_level_MeanReciprocalRank.png" width="600"/>





## Evaluation with Customized Features

### Data Preparation

ExplainaBoard also allows users to customize features, specifically to provide your own bucketing features, submit a system output containing a declaration of your user-defined features (their names, data types, and number of buckets), along with your predictions on test examples. Make sure each test example contains a key for each feature defined in your configuration. Refer to the following example:

```json
{
    "user_defined_features_configs": {
        "rel_type": {
                "dtype": "string",
                "description": "symmetric or asymmetric",
                "num_buckets": 2
        }
    },
    "predictions": {
        "1": {
            "gold_head": "/m/08966",
            "gold_predicate": "/travel/travel_destination/climate./travel/travel_destination_monthly_climate/month",
            "gold_tail": "/m/05lf_",
            "predict": "tail",
            "predictions": [
                "/m/05lf_",
                "/m/02x_y",
                "/m/01nv4h",
                "/m/02l6h",
                "/m/0kz1h"
            ],
            "rel_type": "asymmetric"
        },
        "2": {
            "gold_head": "/m/01hww_",
            "gold_predicate": "/music/performance_role/regular_performances./music/group_membership/group",
            "gold_tail": "/m/01q99h",
            "predict": "tail",
            "predictions": [
                "/m/05563d",
                "/m/02vnpv",
                "/m/02r1tx7",
                "/m/017lb_",
                "/m/03c3yf"
            ],
            "rel_type": "asymmetric"
        },
      ...
```

### Perform Analysis with CLI

An example system output is [provided](https://github.com/neulab/ExplainaBoard/blob/main/explainaboard/tests/artifacts/test-kg-prediction-user-defined-new.json), and you can test it using the following command:

```shell
    explainaboard --task kg-link-tail-prediction --system_outputs ./data/system_outputs/fb15k-237/test-kg-prediction-user-defined-new.json --dataset fb15k_237 > report.json

or

    explainaboard --task kg-link-tail-prediction --system_outputs ./data/system_outputs/fb15k-237/test-kg-prediction-user-defined-new.json > report.json
```


## Advanced Usage
Intead of ExplainaBoard CLI, users could explore more functionality by using 
pythonic interface provided by ExplainaBoard.


### Customized Bucket Order
In some situation, users aim to specify the bucket order according to their needs. Following [code](https://github.com/neulab/ExplainaBoard/blob/8ccd1a71531bc3b9e2f9e539cb001353cc49ebca/explainaboard/tests/test_kg_link_tail_prediction.py#L60) gives an example.



```python

        from explainaboard import TaskType, get_custom_dataset_loader, get_processor, get_datalab_loader
        from explainaboard.loaders.file_loader import DatalabLoaderOption
        from explainaboard.constants import Source
        from explainaboard import (
            FileType,
            get_custom_dataset_loader,
            get_datalab_loader,
            get_pairwise_performance_gap,
            get_processor,
            TaskType,
        )
        from explainaboard.metric import HitsConfig

        # tips: `artifacts_path` is located at: ExplainaBoard/explainaboard/tests/artifacts
        dataset = "./explainaboard/tests/artifacts/kg_link_tail_prediction/no_custom_feature.json
        loader = get_custom_dataset_loader(
            TaskType.kg_link_tail_prediction,
            dataset,
            dataset,
        )
        data = loader.load()
        self.assertEqual(loader.user_defined_features_configs, {})

        metadata = {
            "task_name": TaskType.kg_link_tail_prediction.value,
            "dataset_name": "fb15k-237",
            "metric_names": ["Hits"],
            "sort_by": "performance_value",
            "sort_by_metric": "first",
            "sort_ascending": False,
        }

        processor = get_processor(TaskType.kg_link_tail_prediction.value)
        sys_info = processor.process(metadata, data)
```
The options for the `"sort_by"` option are:
1. `"key"` (default): sort by the bucket's lower boundary, alphabetically, low-to-high.
2. `"performance_value"`: sort by bucket performance. Since each bucket has multiple metrics associated with it, use the `"sort_by_metric"` to choose which metric to sort on.
3. `"n_bucket_samples"`, sort by the number of samples in each bucket.

The `"sort_by_metric"` option is applicable when the `"sort_by"` option is set to `"performance_value"`. The options for the `"sort_by_metric"` option are:
1. `"Hits"`, `"MeanRank"`, `"MeanReciprocalRank"`, etc: sort by a specific metric name.
2. `"first"` (default): sort by the value of the first BucketPerformance object which Explainaboard internally uses, whichever that may be. Not recommended to use this option; instead, specify the metric to sort on explicitly.

The `"sort_by_metric"` option is applicable when the `"sort_by"` option is set to either `"performance_value"` or `"n_bucket_samples"`. The options for the `"sort_ascending"` option are:
1. `False` (default): sort high-to-low.
2. `True`: sort low-to-high; useful for e.g. the `"MeanRank"` metric.

### Customized Hits K
The value of K in `Hits` metric could also be specified by users when needed. Below is an example of how to use this configuration while performing bucket sorting by bucket size:

```python

        from explainaboard import TaskType, get_custom_dataset_loader, get_processor, get_datalab_loader
        from explainaboard.loaders.file_loader import DatalabLoaderOption
        from explainaboard.constants import Source
        from explainaboard import (
            FileType,
            get_custom_dataset_loader,
            get_datalab_loader,
            get_pairwise_performance_gap,
            get_processor,
            TaskType,
        )
        from explainaboard.metric import HitsConfig

        dataset = "explainaboard/tests/artifacts/kg_link_tail_prediction/no_custom_feature.json
        loader = get_custom_dataset_loader(
            TaskType.kg_link_tail_prediction,
            dataset,
            dataset,
            Source.local_filesystem,
            Source.local_filesystem,
            FileType.json,
            FileType.json,
        )
        data = loader.load()

        metadata = {
            "task_name": TaskType.kg_link_tail_prediction.value,
            "dataset_name": "fb15k-237-subset",
            "metric_names": ["Hits"],
            "metric_configs": {"Hits": HitsConfig(hits_k=4)},  # you can modify k here
            "sort_by": "n_bucket_samples",
            "sort_ascending": False,  # buckets with many samples appear first
        }

        processor = get_processor(TaskType.kg_link_tail_prediction.value)

        sys_info = processor.process(metadata, data)

        # analysis.write_to_directory("./")
```


### Record Other System Detailed Information

The basic idea is that users can specify other system-related information (e.g., hyper-parameters)
via adding a key-value into `metadata`
```python
        metadata = {
            "task_name": TaskType.text_classification.value,
            "metric_names": ["Accuracy"],
            "system_details": system_details,
        }
```
[Here](https://github.com/neulab/ExplainaBoard/blob/main/explainaboard/tests/test_system_details.py) is a complete code.
