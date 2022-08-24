# Programmatic Interface

Below is some example code that demonstrates how you can access ExplainaBoard programmatically.
You can see more about the supported tasks and data formats in 

You can process an existing dataset, such as `sst2`:

```python
from explainaboard import TaskType, get_loader_class, get_processor
# using a dataset we provide (datalab)
loader = get_loader_class(TaskType.text_classification).from_datalab(
    dataset=DatalabLoaderOption("sst2"),
    output_data="./integration_tests/artifacts/text_classification/output_sst2.txt",
    output_source=Source.local_filesystem,
    output_file_type=FileType.text,
)
data = loader.load().samples
processor = get_processor(TaskType.text_classification)
analysis = processor.process(metadata={}, sys_output=data)
analysis.write_to_directory("./")
```

or use a custom dataset input as a raw file

```python
dataset = "./integration_tests/artifacts/summarization/dataset.tsv"
output = "./integration_tests/artifacts/summarization/output.tsv"
loader = get_loader_class(TaskType.summarization)(dataset_data=dataset, output_data=output)
data = loader.load()
processor = get_processor(TaskType.summarization)
analysis = processor.process(metadata={}, sys_output=data)
analysis.write_to_directory("./")
```
