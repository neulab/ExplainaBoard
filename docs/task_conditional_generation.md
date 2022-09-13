# Analyzing Conditional Text Generation Tasks

Before diving into the detail of this doc, you're strongly recommended to know [some
important concepts about system analyses](concepts_about_system_analysis.md).


Conditional text generation is a class of tasks where you generate text based on some conditioning context.
This can include a wide variety of tasks, such as:

* **Text Summarization:** generates a summary *y* given an input document *x*.
  An example dataset may be [CNN/Daily Mail](http://datalab.nlpedia.ai/#/normal_dataset/6176883933e51a7edda9dd68/dataset_metadata).
* **Machine Translation:** generates a text *y* in one language given an input text *x* in another language.
  An example dataset may be the [TED Multilingual Dataset](https://huggingface.co/datasets/ted_multi).
* **Code Generation:** generates a program *y* in a programming language such as Python given an input command *x* in natural language.
  An example dataset may be the [CoNaLa](https://conala-corpus.github.io/) English to Python generation dataset.


## Data Preparation

 
### Format of `Dataset` File


* (1) `datalab`: if your datasets have been supported by [datalab](https://github.com/ExpressAI/DataLab/tree/main/datasets),
    you fortunately don't need to prepare the dataset. 

* (2) `tsv` (without column names at the first row), see one [example](https://github.com/neulab/ExplainaBoard/blob/main/data/system_outputs/cnndm/cnndm_mini-dataset.tsv)
```python
This is a good movie    这是一部好电影
...
```
where the first column represents source text and the 2nd column denotes gold reference.

* (3) `json` (basically, it's a list of dictionaries with two keys: `source` and `reference`)
```json
[
  {"source": "This is a good movie", "reference": "这是一部好电影"},
  ...
]
```




### Format of `System Output` File
In this task, your system outputs should be as follows:


```
predicted_output_text
```

Here is an example system output file for summarization on a subset of the CNN/Daily Mail articles:
* [cnndm_mini-bart-output.txt](https://github.com/neulab/ExplainaBoard/blob/main/data/system_outputs/cnndm/cnndm_mini-bart-output.txt) 

And here are two examples for machine translation from Slovak to English, an NMT and phrase-based MT system:
* [ted_multi_slk_eng-nmt-output.txt](https://github.com/neulab/ExplainaBoard/blob/main/data/system_outputs/ted_multi/ted_multi_slk_eng-nmt-output.txt) 
* [ted_multi_slk_eng-pbmt-output.txt](https://github.com/neulab/ExplainaBoard/blob/main/data/system_outputs/ted_multi/ted_multi_slk_eng-pbmt-output.txt) 

Here is an example output for code generation. Note that this is in JSON format, and specifically specifies Python as the output language.
This is important so the code is tokenized properly during evaluation.
* [conala-baseline-output.json](https://github.com/neulab/ExplainaBoard/blob/main/data/system_outputs/conala/conala-baseline-output.json)

## Performing Basic Analysis on Summarization

The preferred method of doing analysis is to load the dataset from DataLab.
You can load the`cnn_dailymail` dataset but because the test set is large we don't
include it directly in the explainaboard repository, but you can get an example by
downloading with wget:
```shell
wget -P ./data/system_outputs/cnndm/ https://storage.googleapis.com/inspired-public-data/explainaboard/task_data/summarization/cnndm-bart-output.txt
```

Then run the below command and it should work:
```shell
explainaboard --task summarization --dataset cnn_dailymail --system-outputs ./data/system_outputs/cnndm/cnndm-bart-output.txt --metrics rouge2
```

* `--task`: denotes the task name.
* `--system-outputs`: denote the path of system outputs. Multiple one should be
  separated by space, for example, system1 system2
* `--dataset`: optional, denotes the dataset name
* `--metrics`: optional, different metrics should be separated by space. See [more supported metrics](https://github.com/neulab/ExplainaBoard/blob/main/docs/supported_tasks.md#summarization)
* `report.json`: the generated analysis file with json format. Tips: you can use a json viewer
  like [this one](http://jsonviewer.stack.hu/) or Python's `python -m json.tool` to convert
  the JSON into a prettified and readable format.

In addition, you can use a custom dataset, in which case the format should be
```
source_sentence \t target_sentence
```

In this case, we can directly use the miniature dataset distributed with the repo:
```shell
explainaboard --task summarization --custom-dataset-paths ./data/system_outputs/cnndm/cnndm_mini-dataset.tsv --system-outputs ./data/system_outputs/cnndm/cnndm_mini-bart-output.txt --metrics rouge2 bart_score_en_ref
```

## Other Task Examples

### Machine Translation

Try it out for translation as below. The examples use a custom dataset that is not included in DataLab at the moment.
```shell
explainaboard --task machine-translation --custom-dataset-paths ./data/system_outputs/ted_multi/ted_multi_slk_eng-dataset.tsv --system-outputs ./data/system_outputs/ted_multi/ted_multi_slk_eng-nmt-output.txt --metrics bleu comet
```
Note 1: The number of `--custom-dataset-paths` need to match the number of `system-outputs`

Note 2: If you want to perform **pair-wise analysis** for two system outputs on the same reference, you can pass in the paths separated by space, for example, `--system-outputs system1 system2`. Please also pass in the same paths twice for `--custom-dataset-paths`, for example, `--custom-dataset-paths path1 path1`. 

### Code Generation

You can try out evaluation of code generation on the CoNaLa dataset in DataLab as below:
```shell
explainaboard --task machine-translation --dataset conala --output-file-type json --system-outputs ./data/system_outputs/conala/conala-baseline-output.json --report-json report.json
```

You can also use a custom code generation dataset:
```shell
explainaboard --task machine-translation --custom-dataset-file-type json --custom-dataset-paths data/system_outputs/conala/conala-dataset.json --output-file-type json --system-outputs ./data/system_outputs/conala/conala-baseline-output.json --report-json report.json
```

## Notes

**Other Conditional Text Generation Tasks:** You can probably also get a start on
analyzing other sequence-to-sequence tasks (e.g. text style transfer) by just specifying
`machine-translation` or `summarization` and feeding in the data from your dataset.
This would give you a start, but you may want to design other features that are specific
for this  task. If you'd like help with this, feel free to open an issue!

**Multi-document Summarization:** ExplainaBoard supports single-document summarization
and text compression, but not multi-document summarization or other similar tasks like
retrieval-based QA.  We would welcome help with adding support for this, so similarly open an issue!
