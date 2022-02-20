# Analyzing Conditional Text Generation Tasks

Conditional text generation is a class of tasks where you generate text based on some conditioning context.
This can include a wide variety of tasks, such as:

* **Text Summarization:** generates a summary *y* given an input document *x*.
  An example dataset may be [CNN/Daily Mail](http://datalab.nlpedia.ai/#/normal_dataset/6176883933e51a7edda9dd68/dataset_metadata).
* **Machine Translation:** generates a text *y* in one language given an input text *x* in another language.
  An example dataset may be the [TED Multilingual Dataset](https://huggingface.co/datasets/ted_multi).

## Data Preparation

In order to perform analysis of your results, they should be in the following
TSV format:

```
input_text \t reference_output_text \t predicted_output_text
```

Here is an example system output file for summarization of CNN/Daily Mail articles:
* [cnndm_mini.bart](https://github.com/neulab/ExplainaBoard/blob/main/data/system_outputs/cnndm/cnndm_mini.bart) 

And here are two examples for machine translation from Slovak to English:
* [ted_multi_slk_eng.pbmt](https://github.com/neulab/ExplainaBoard/blob/main/data/system_outputs/ted_multi/ted_multi_slk_eng.pbmt) 
* [ted_multi_slk_eng.nmt](https://github.com/neulab/ExplainaBoard/blob/main/data/system_outputs/ted_multi/ted_multi_slk_eng.nmt) 



## Performing Basic Analysis

In order to perform your basic analysis, you can run the following commands for summarization and MT respectively:

```shell
explainaboard --task summarization --system_outputs ./data/system_outputs/cnndm/cnndm_mini.bart --metrics bart_score_summ rouge2
```

```shell
explainaboard --task machine-translation --system_outputs ./data/system_outputs/ted_multi/ted_multi_slk_eng.nmt --metrics bleu
```

where
* `--task`: denotes the task name. 
* `--system_outputs`: denote the path of system outputs. Multiple one should be 
  separated by space, for example, system1 system2
* `--dataset`: optional, denotes the dataset name
* `--metrics`: optional, different metrics should be separated by space. See [more supported metrics](https://github.com/neulab/ExplainaBoard/blob/main/docs/supported_tasks.md#summarization)
* `report.json`: the generated analysis file with json format. Tips: you can use a json viewer
                  like [this one](http://jsonviewer.stack.hu/) or Python's `python -m json.tool` to convert
                  the JSON into a prettified and readable format.



## Bucketing Features
* `source_len`: the length of the source document
* `compression`: the compression ratio `len(src)/len(ref)`
* [`copy_len`](https://aclanthology.org/2020.findings-emnlp.329.pdf): measures the average length of segments in summary copied from source document.
* [`coverage`](https://aclanthology.org/2020.findings-emnlp.329.pdf): illustrates the overlap rate between document and summary, it is defined as the proportion of the copied segments in
summary.
* [`novelty`]((https://aclanthology.org/2020.findings-emnlp.329.pdf)): is defined as the proportion of segments in the summaries that haven’t
appeared in source documents. The segments is instantiated as 2-grams.

## Notes

**Other Conditional Text Generation Tasks:** You can probably also get a start on analyzing other sequence-to-sequence
tasks (e.g. text style transfer) by just specifying `machine-translation` or `summarization` and feeding in the data
from your dataset. This would give you a start, but you may want to design other features that are specific for this
task. If you'd like help with this, feel free to open an issue!

**Multi-document Summarization:** ExplainaBoard supports single-document summarization and text compression, but not
multi-document summarization or other similar tasks like retrieval-based QA.
We would welcome help with adding support for this, so similarly open an issue!