# Add New Format


## Case 1: Supported Formats
If the format of your datasets has already been supported
by existing library, directly using it without any library-level
modification

* `TaskType.text_classification`
  * `FileType.tsv`
* `TaskType.named_entity_recognition`
  * `FileType.conll`
* `TaskType.summarization`
  * `FileType.tsv`
* `TaskType.extractive_qa`
  * `FileType.json` (same format with squad)
  

For example, supposing you have a system output of summarization task
with  `tsv` format:

```python
from explainaboard import TaskType, get_loader, get_processor

path_data = "./artifacts/test-summ.tsv"
loader = get_loader(task = TaskType.summarization, 
                    file_type = FileType.tsv,
                    data = path_data)
data = loader.load()
processor = get_processor(TaskType.summarization, data = data)
analysis = processor.process()
analysis.write_to_directory("./")
```


## Case 2: Unsupported Formats
If your dataset is in a new format that current SDK doesn't support, you can
* (1) reformat your data into the format that current library support
* (2) you can also re-write the `loader.load()` function to make it 
  support your format.
  Taking summarization task for example, supposing existing SDK only supports
  `tsv` format, we can make `json` format supported by add following code inside
  `loaders.summarization.TextSummarizationLoader.loader()`
  ```python
      def load(self) -> Iterable[Dict]:
        raw_data = self._load_raw_data_points()
        data: List[Dict] = []
        if self._file_type == FileType.tsv:
            for id, dp in enumerate(raw_data):
                source, reference, hypothesis = dp[:3]
                data.append({"id": id,
                             "source": source.strip(),
                             "reference": reference.strip(),
                             "hypothesis": hypothesis.strip()})
        if self._file_type == FileType.json: # This function has been unittested
            for id, info in enumerate(raw_data):
                source, reference, hypothesis = info["source"], info["references"], info["hypothesis"]
                data.append({"id": id,
                             "source": source.strip(),
                             "reference": reference.strip(),
                             "hypothesis": hypothesis.strip()})
        else:
            raise NotImplementedError
        return data
```
  