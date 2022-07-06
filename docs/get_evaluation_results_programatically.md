# How to Evaluate your Models Programmatically?


This doc details 
* how to evaluate your systems using ExplainaBoard programmatically
* how to collect different results 





## Evaluation

Take the `kg-link-tail-prediction` task, for example, by running the following code,
all analysis information will be stored in `sys_info.`


```python
from explainaboard import TaskType, get_custom_dataset_loader, get_processor

# Load the data
dataset = "./explainaboard/tests/artifacts/kg_link_tail_prediction/no_custom_feature.json"
task = TaskType.kg_link_tail_prediction
loader = get_custom_dataset_loader(task, dataset, dataset)
data = loader.load()
# Initialize the processor and perform the processing
processor = get_processor(TaskType.kg_link_tail_prediction.value)
sys_info = processor.process(metadata={}, sys_output=data.samples)
```


## Manipulate Analysis Results
The above code conducts the evaluation and puts everything in `sys_info.` In what follows,
we will see how different types of information from `sys_info` are collected.


#### Print Bucket-wise Evaluation Results

```python
   processor.print_analyses(sys_info.results.analyses)
```


#### Save analysis report locally
```python
sys_info.print_as_json(file=open("./report.json", 'w'))
```

Here is an [example](https://github.com/neulab/ExplainaBoard/blob/86d96b83d5ebf60adbdbdaa3a00883546fa05fde/data/reports/report_kg.json).


#### Get overall results of different metrics
```python
for metric_name, metric_info in sys_info.results.overall.items():
    metric_name = metric_info.metric_name
    value = metric_info.value
    confidence_score_low = metric_info.confidence_score_low
    confidence_score_high = metric_info.confidence_score_high
```



#### Get fine-grained results
```python
for feature_name, feature_info in sys_info.results.analyses.items():
    for bucket_name, bucket_info in feature_info.items():
        bucket_n_samples = bucket_info.n_samples
        for bucket_performance in bucket_info.performances:
            metric_name = bucket_performance.metric_name
            value = bucket_performance.value
            confidence_score_low = bucket_performance.confidence_score_low
            confidence_score_high = bucket_performance.confidence_score_high

            print("------------------------------------------------------")
            print(f"feature_name:{feature_name} bucket_name:{bucket_name}")
            print(f"metric_name:{metric_name}\n"
                  f"value:{value}\n"
                  f"confidence_score_low:{confidence_score_low}\n"
                  f"confidence_score_high:{confidence_score_high}\n")
```


Note: the full version of the code in this code can be found [here](https://github.com/neulab/ExplainaBoard/blob/add_demo_example_kg/docs/example_scripts/test_kg.py)

