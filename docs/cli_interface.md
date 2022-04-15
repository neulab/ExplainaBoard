# CLI Interface to Various Tasks

Below is a (mostly complete) list of tasks that ExplainaBoard currently supports, along with examples of how to analyze different tasks.
In particular [text classification](#text-classification) is a good example to start with.

**General notes:**
* Click the link on the task name for more details, or when no link exists you can open the example data to see what the file format looks like.
* You can either analyze an existing dataset included in [Datalab](https://github.com/expressai/datalab) or use your own custom dataset. The directions below describe how to do both in most cases.
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

The below example loads a miniature version of the CNN-daily mail dataset (100 lines only) from an existing file:
```shell
explainaboard --task summarization --custom_dataset_paths ./data/system_outputs/cnndm/cnndm_mini-dataset.tsv --system_outputs ./data/system_outputs/cnndm/cnndm_mini-bart-output.txt --metrics rouge2 bart_score_en_ref
```
Note that this uses two different metrics separated by a space.

You could also load the `cnn_dailymail` dataset from DataLab, but because the test set is large we don't include it directly in the explainaboard repository. Replace MY_FILE below with an output from your system and it should work:
```shell
explainaboard --task summarization --dataset cnn_dailymail --system_outputs MY_FILE --metrics rouge2 bart_score_summ
```


## Named Entity Recognition

Named entity recognition recognizes entities such as people, organizations, or locations in text.
The below examples demonstrate how you can perform such analysis on the CoNLL 2003 English named entity recognition dataset.

**CLI Example**

The below example loads the `conll2003` NER dataset from DataLab:
```shell
explainaboard --task named-entity-recognition --dataset conll2003 --sub_dataset ner --system_outputs ./data/system_outputs/conll2003/conll2003-elmo-output.conll
```

Alternatively, you can reference a dataset file directly.
```shell
explainaboard --task named-entity-recognition --custom_dataset_paths ./data/system_outputs/conll2003/conll2003-dataset.conll --system_outputs ./data/system_outputs/conll2003/conll2003-elmo-output.conll 
```


## [Extractive QA](task_extractive_qa.md)

Extractive QA attempts to answer queries based on extracting segments from an evidence passage.
The below example performs this extraction on the dataset SQuAD.

**CLI Example**

Below is an example of referencing the dataset directly.
```shell
explainaboard --task question-answering-extractive --custom_dataset_paths ./data/system_outputs/squad/squad_mini-dataset.json --system_outputs ./data/system_outputs/squad/squad_mini-example-output.json > report.json
```

The below example loads the `squad` dataset from DataLab. There is an [open issue](https://github.com/neulab/ExplainaBoard/issues/239) that prevents the specification of a dataset split, so this will not work at the moment. But we are working on it.
```shell
explainaboard --task question-answering-extractive --dataset squad --system_outputs MY_FILE > report.json
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
