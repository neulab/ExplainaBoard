# CLI Interface Currently Supported Tasks

Below is a (somewhat incomplete) list of tasks that ExplainaBoard currently supports, along with examples of how to analyze different tasks.
In particular [text classification](#text-classification) is a good example to start with.

**General notes:**
* Click the link on the task name for more details, or when no link exists you can open the example data to see what the file format looks like.
* You can either analyze an existing dataset included in [Datalab](https://github.com/expressai/datalab) or use your own custom dataset. The directions below describe how to do both.
* All of the examples below will output a json report to standard out, and you can use Python's pretty printing tool to see it in a more readable format (add `| python -m json.tool` at the end of any of the lines).


We welcome contributions of [more tasks](add_new_tasks.md), or detailed documentation for tasks where the documentation does not yet exist! Please open an issue or file a PR.

**Table of Contents**
* [Text Classification](#text-classification)
* [Text Pair Classification](#text-pair-classification)
* [Conditional Text Generation](#conditional-generation)
* [Named Entity Recognition](#named-entity-recognition)
* [Extractive QA](#extractive-qa-extractive)
* [Multiple Choice QA](#multiple-choice-qa)
* [Aspect-based Sentiment Classification](#aspect-based-sentiment-classification)
* [KG Link Tail Prediction](#kg-link-tail-prediction)


## [Text Classification](task_text_classification.md)

Text classification consists of classifying text into different categories, such as sentiment values or topics.
The below example performs an analysis on the Stanford Sentiment Treebank, a set of sentiment tags over English reviews.

**CLI Examples**

The below example loads the `sst2` dataset from DataLab:
```shell
explainaboard --task text-classification --dataset sst2 --system_outputs ./data/system_outputs/sst2/sst2-lstm-output.txt
```

The below example loads a dataset from an existing file:
```shell
explainaboard --task text-classification --custom_dataset_paths ./data/system_outputs/sst2/sst2-dataset.tsv --system_outputs ./data/system_outputs/sst2/sst2-lstm-output.txt
```


## [Text Pair Classification](task_text_pair_classification.md)

Classification of pairs of text, such as natural language inference or paraphrase detection.
The example below concerns natural language infernce, predicting whether a premise, entails, contradicts, or is neutral with respect to a hypothesis, on the Stanford Natural Language Inference dataset.

**CLI Example**

The below example loads the `snli` dataset from DataLab:
```shell
explainaboard --task text-pair-classification --dataset snli --system_outputs ./data/system_outputs/snli/snli-bert-output.txt
```

The below example loads a dataset from an existing file:
```shell
explainaboard --task text-pair-classification --custom_dataset_paths ./data/system_outputs/snli/snli-dataset.tsv --system_outputs ./data/system_outputs/snli/snli-bert-output.txt
```


## [Conditional Text Generation](task_conditional_generation.md)

Conditional text generation concerns generation of one text based on other texts, including tasks like summarization and machine translation.
The below example evaluates a summarization system on the CNN-daily mail dataset.

**CLI Example**
```shell
explainaboard --task summarization --system_outputs ./data/system_outputs/cnndm/cnndm_mini.bart --metrics rouge2 bart_score_summ
```
Note that this uses two different metrics separated by a space.


## Named Entity Recognition

Named entity recognition recognizes entities such as people, organizations, or locations in text.
The below examples demonstrate how you can perform such analysis on the CoNLL 2003 English named entity recognition dataset.

**CLI Example**
```shell
explainaboard --task named-entity-recognition --system_outputs ./data/system_outputs/conll2003/conll2003.elmo
```

Alternatively, you can reference the dataset directly. 

```
explainaboard --task named-entity-recognition --dataset conll2003 --sub_dataset ner  --system_outputs ./data/system_outputs/conll2003/conll2003.elmo
```


## [Extractive QA](task_extractive_qa.md)

Extractive QA attempts to answer queries based on extracting segments from an evidence passage.
The below example performs this extraction on the dataset SQuAD.

**CLI Example**
```shell
explainaboard --task question-answering-extractive --system_outputs ./data/system_outputs/squad/test-qa-extractive.json > report.json
```


## [Multiple Choice QA](task_qa_multiple_choice.md)

Answer a question from multiple options.
The following example demonstrates this on the metaphor QA dataset.

**CLI Example**
```shell
explainaboard --task qa-multiple-choice --system_outputs ./data/system_outputs/metaphor_qa/gpt2.json > report.json
```


## [KG Link Tail Prediction](task_kg_link_tail_prediction.md)

Predicting the tail entity of missing links in knowledge graphs

**CLI Example**
```shell
    explainaboard --task kg-link-tail-prediction --system_outputs ./data/system_outputs/fb15k-237/test-kg-prediction-no-user-defined-new.json > report.json
    # or
    explainaboard --task kg-link-tail-prediction --system_outputs ./data/system_outputs/fb15k-237/test-kg-prediction-no-user-defined-new.json --dataset fb15k_237 > report.json
```
 

## [Aspect-based Sentiment Classification](task_aspect_based_sentiment_classification.md)

Predict the sentiment of a text based on a specific aspect.

**CLI Example**
```shell
explainaboard --task aspect-based-sentiment-classification --system_outputs ./data/system_outputs/absa/test-aspect.tsv > ./data/reports/report_absa.json
```
