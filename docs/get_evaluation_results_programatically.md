# How to Evaluate your Models Programmatically?


This doc details 
* how to evaluate your systems using ExplainaBoard programmatically
* how to collect different results 





## Evaluation

Take the `kg-link-tail-prediction` task, for example, by running the following code,
all analysis information will be stored in `sys_info.`


```python
from explainaboard import TaskType, get_loader_class, get_processor

# Load the data
dataset = "./explainaboard/tests/artifacts/kg_link_tail_prediction/no_custom_feature.json"
task = TaskType.kg_link_tail_prediction
loader = get_loader_class(task)(dataset, dataset)
data = loader.load()
# Initialize the processor and perform the processing
processor = get_processor(TaskType.kg_link_tail_prediction.value)
sys_info = processor.process(metadata={}, sys_output=data.samples)
```


## Manipulate Analysis Results
The above code conducts the evaluation and puts everything in `sys_info.` In what follows,
we will see how different types of information from `sys_info` are collected.


#### Save analysis report locally
```python
sys_info.print_as_json(file=open("./report.json", 'w'))
```

Here is an [example](https://github.com/neulab/ExplainaBoard/blob/86d96b83d5ebf60adbdbdaa3a00883546fa05fde/data/reports/report_kg.json).


#### Get overall results of different metrics
```python
for overall_level in sys_info.results.overall:
    for metric_stat in overall_level:
        print(f'{metric_stat.metric_name}\t{metric_stat.value}')
```

#### Print analysis results

You can also print fine-grained analyses:

```python
for analysis_level in sys_info.results.analyses:
    for analysis in analysis_level:
        if analysis is not None:
            analysis.print()
```