# Concepts about System Analysis
In this docs, we will explain some important concepts that
you need to know before performing system analyses using Explainaboard.

## What a "dataset" and "system output" is
To perform an analysis of your results, usually, two types of files should be prepared: `dataset`
and `system output` 

### `Dataset`
`Dataset` usually consists of test samples together with true labels (or references in text generation
tasks).
For example, in text classification, the `dataset` could be organized in a tsv format:
```text
I love this movie   positive
The movie is too long   negative
...
``` 

### `System output`
`System output` is frequently composed of predicted labels (or hypotheses, e.g., system-generated text),
but sometimes `system output` will also contain test samples, such as `CoNLL` format in sequence labeling tasks.
For example, in text classification, the `system output` could be organized in a text format:
```text
positive
negative
...
```



## How the dataset can be gotten from two sources, either "datalab" or "custom"
The `dataset` typically could be obtained from two sources (so far): one is supported by [Datalab](https://aclanthology.org/2022.acl-demo.18.pdf) (Xiao et al 2022) and the
other is custom.
* if your datasets have been supported by [datalab](https://github.com/ExpressAI/DataLab/tree/main/datasets),
you, fortunately, don't need to prepare the dataset. Instead, you just need to get the dataset name for later use.
For example, the following command can be used to perform system analysis on `sst2` dataset (supported by datalab)
```shell script
explainaboard --task text-classification --dataset sst2 --system-outputs ./data/system_outputs/sst2/sst2-lstm-output.txt
```
If the dataset hasn't been supported by datalab, we then need to specify a file path:
```shell script
explainaboard --task text-classification --custom-dataset-paths ./data/system_outputs/sst2/sst2-dataset.tsv --system-outputs ./data/system_outputs/sst2/sst2-lstm-output.txt
```

* if your datasets haven't been supported by datalab but you want them supported, you can follow this 
[doc](https://github.com/ExpressAI/DataLab/blob/main/docs/SDK/add_new_datasets_into_sdk.md) to add them.




## How different formats (e.g. json, tsv, etc.) are supported
In ExplainaBoard, users are allowed to prepare custom datasets with different formats, such as
`json`, `tsv` etc. We will detail this in each task's description.