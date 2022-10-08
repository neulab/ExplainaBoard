# CLI Interface to Various Tasks

Below is a (mostly complete) list of tasks that ExplainaBoard currently supports, along
with examples of how to analyze different tasks. In particular
[text classification](#text-classification) is a good example to start with.

**General notes:**

* Click the link on the task name for more details, or when no link exists you can open
  the example data to see what the file format looks like.
* You can either analyze an existing dataset included in
  [Datalab](https://github.com/expressai/datalab) or use your own custom dataset.
  The directions below describe how to do both in most cases, but using DataLab has some
  advantages such as allowing for easy calculation of training-set features and
  compatibility with ExplainaBoard online leaderboards. You can check the list of
  [datasets supported in DataLab](https://github.com/ExpressAI/DataLab/tree/main/datasets)
  and [add your dataset](https://github.com/ExpressAI/DataLab/blob/main/docs/SDK/add_new_datasets_into_sdk.md)
  if it doesn't exist.
* All of the examples below will output a json report to standard out, which you can
  pipe to a file such as `report.json` for later use. Also, check out our
  [visualization tools](visualization.md).

We welcome contributions of [more tasks](add_new_tasks.md), or detailed documentation
for tasks where the documentation does not yet exist! Please open an issue or file a PR.

**Table of Contents**

* [Text Classification](#text-classification)
* [Text Pair Classification](#text-pair-classification)
* [Conditional Text Generation](#conditional-text-generation)
* [Language Modeling](#language-modeling)
* [Named Entity Recognition](#named-entity-recognition)
* [Word Segmentation](#word-segmentation)
* [Chunking](#chunking)
* [Extractive QA](#extractive-qa)
* [Multiple Choice QA](#multiple-choice-qa)
* [Hybrid Table Text QA](#hybrid-table-text-qa)
* [Aspect-based Sentiment Classification](#aspect-based-sentiment-classification)
* [KG Link Tail Prediction](#kg-link-tail-prediction)
* [Multiple choice Cloze](#multiple-choice-cloze)
* [Generative Cloze](#generative-cloze)
* [Grammatical Error Correction](#grammatical-error-correction)
* [Tabular Classification](#tabular-classification)
* [Tabular Regression](#tabular-regression)
* [Argument Pair Extraction](argument-pair-extraction)
* [Argument Pair Identification](argument-pair-identification)
* [WMT Metrics Direct Assessment Meta-evaluation](#wmt-metrics-direct-assessment-meta-evaluation)

## [Text Classification](task_text_classification.md)

Text classification consists of classifying text into different categories, such as
sentiment values or topics. The below example performs an analysis on the Stanford
Sentiment Treebank, a set of sentiment tags over English reviews.

**CLI Examples**

The below example loads the `sst2` dataset from DataLab:

```shell
explainaboard --task text-classification --dataset sst2 --system-outputs ./data/system_outputs/sst2/sst2-lstm-output.txt
```

The below example loads a dataset from an existing file:

```shell
explainaboard --task text-classification --custom-dataset-paths ./data/system_outputs/sst2/sst2-dataset.tsv --system-outputs ./data/system_outputs/sst2/sst2-lstm-output.txt
```

## [Text Pair Classification](task_text_pair_classification.md)

Classification of pairs of text, such as natural language inference or paraphrase
detection. The example below concerns natural language infernce, predicting whether a
premise, entails, contradicts, or is neutral with respect to a hypothesis, on the
Stanford Natural Language Inference dataset.

**CLI Example**

The below example loads the `snli` dataset from DataLab:

```shell
explainaboard --task text-pair-classification --dataset snli --system-outputs ./data/system_outputs/snli/snli-roberta-output.txt
```

The below example loads a dataset from an existing file:

```shell
explainaboard --task text-pair-classification --custom-dataset-paths ./data/system_outputs/snli/snli-dataset.tsv --system-outputs ./data/system_outputs/snli/snli-roberta-output.txt
```

## [Conditional Text Generation](task_conditional_generation.md)

Conditional text generation concerns generation of one text based on other texts,
including tasks like summarization and machine translation. The below example evaluates
a summarization system on the CNN-daily mail dataset.

**CLI Example**

The below example loads a miniature version of the CNN-daily mail dataset (100 lines only) from an existing file:

```shell
explainaboard --task summarization --custom-dataset-paths ./data/system_outputs/cnndm/cnndm_mini-dataset.tsv --system-outputs ./data/system_outputs/cnndm/cnndm_mini-bart-output.txt --metrics rouge2 bart_score_en_ref
```

Note that this uses two different metrics separated by a space.

You could also load the `cnn_dailymail` dataset from DataLab.
Because the test set is large we don't include it directly in the explainaboard repository, but you can get an example by downloading with wget:

```shell
wget -P ./data/system_outputs/cnndm/ https://storage.googleapis.com/inspired-public-data/explainaboard/task_data/summarization/cnndm-bart-output.txt
```

Then run the below command and it should work:

```shell
explainaboard --task summarization --dataset cnn_dailymail --system-outputs ./data/system_outputs/cnndm/cnndm-bart-output.txt --metrics rouge2
```

## Language Modeling

Language modeling is the task of predicting the probability for words in a text.
You can analyze your language model outputs by inputting a file that has one log
probability for each space-separated word. Here is an example:

**CLI Example**

The below example analyzes the wikitext corpus:

```shell
explainaboard --task language-modeling --custom-dataset-paths ./data/system_outputs/wikitext/wikitext-dataset.txt --system-outputs ./data/system_outputs/wikitext-sys1-output.txt
```

## Named Entity Recognition

Named entity recognition recognizes entities such as people, organizations, or locations in text.
The below examples demonstrate how you can perform such analysis on the CoNLL 2003 English named entity recognition dataset.

**CLI Example**

The below example loads the `conll2003` NER dataset from DataLab:

```shell
explainaboard --task named-entity-recognition --dataset conll2003 --sub-dataset ner --system-outputs ./data/system_outputs/conll2003/conll2003-elmo-output.conll
```

Alternatively, you can reference a dataset file directly.

```shell
explainaboard --task named-entity-recognition --custom-dataset-paths ./data/system_outputs/conll2003/conll2003-dataset.conll --system-outputs ./data/system_outputs/conll2003/conll2003-elmo-output.conll 
```

## Word Segmentation

Word segmentation aims to segment texts without spaces between words.

**CLI Example**

The below example loads the `msr` dataset from DataLab:

```shell
explainaboard --task word-segmentation --dataset msr --system-outputs ./data/system_outputs/cws/test-msr-predictions.tsv
```

Note that the file `test-msr-predictions.tsv` can be downloaded [here](https://datalab-hub.s3.amazonaws.com/predictions/test-msr-predictions.tsv)

Alternatively, you can reference a dataset file directly.

```
explainaboard --task word-segmentation --custom-dataset-paths ./data/system_outputs/cws/test.tsv --system-outputs ./data/system_outputs/cws/prediction.tsv
```

## Chunking

Dividing text into syntactically related non-overlapping groups of words.

**CLI Example**

The below example loads the `conll00_chunk` dataset from DataLab:

```shell
explainaboard --task chunking --dataset conll00_chunk --system-outputs ./data/system_outputs/chunking/test-conll00-predictions.tsv
```

Alternatively, you can reference a dataset file directly.

```
explainaboard --task chunking --custom-dataset-paths ./data/system_outputs/chunking/dataset-test-conll00.tsv --system-outputs ./data/system_outputs/chunking/test-conll00-predictions.tsv
```

## [Extractive QA](task_extractive_qa.md)

Extractive QA attempts to answer queries based on extracting segments from an evidence passage.
The below example performs this extraction on the dataset SQuAD.

**CLI Example**

Below is an example of referencing the dataset directly.

```shell
explainaboard --task qa-extractive --custom-dataset-paths ./data/system_outputs/squad/squad_mini-dataset.json --system-outputs ./data/system_outputs/squad/squad_mini-example-output.json > report.json
```

The below example loads the `squad` dataset from DataLab. There is an [open issue](https://github.com/neulab/ExplainaBoard/issues/239) that prevents the specification of a dataset split, so this will not work at the moment. But we are working on it.

```shell
explainaboard --task qa-extractive --dataset squad --system-outputs MY_FILE > report.json
```

## [Hybrid Table Text QA](task_qa_table_text_hybrid.md)

This task aims to answer a question based on a hybrid of tabular
and textual context, e.g., [Zhu et al.2021](https://aclanthology.org/2021.acl-long.254.pdf).

**CLI Example**

The below example loads the `tat_qa` dataset from DataLab.

```shell
explainaboard --task qa-tat --output-file-type json --dataset tat_qa --system-outputs predictions_list.json > report.json
```

where you can download the file `predictions_list.json` by:

```shell
wget -P ./ https://explainaboard.s3.amazonaws.com/system_outputs/qa_table_text_hybrid/predictions_list.json
```

## [Open Domain QA](task_qa_open_domain.md)

Open-domain QA aims to answer a question in the form of natural language based on large-scale
unstructured documents

Following examples show how an open-domain QA system can be evaluated with detailed analyses using
ExplainaBoard CLI.

**CLI Example**

Using Build-in datasets from DataLab:

```shell
explainaboard --task qa-open-domain --dataset natural_questions_comp_gen   --system-outputs ./data/system_outputs/qa_open_domain/test.dpr.nq.txt  > report.json
```

## [Multiple Choice QA](task_qa_multiple_choice.md)

Answer a question from multiple options.
The following example demonstrates this on the metaphor QA dataset.

**CLI Example**

The below example loads the `fig_qa` dataset from DataLab.

```shell
explainaboard --task qa-multiple-choice --dataset fig_qa --system-outputs ./data/system_outputs/fig_qa/fig_qa-gptneo-output.json > report.json
```

And this is what it looks like with a custom dataset.

```shell
explainaboard --task qa-multiple-choice --custom-dataset-paths ./data/system_outputs/fig_qa/fig_qa-dataset.json --system-outputs ./data/system_outputs/fig_qa/fig_qa-gptneo-output.json > report.json
```

## [KG Link Tail Prediction](task_kg_link_tail_prediction.md)

Predicting the tail entity of missing links in knowledge graphs

**CLI Example**

The below example loads the `fb15k_237` dataset from DataLab.

```shell
    wget https://datalab-hub.s3.amazonaws.com/predictions/test_distmult.json
    explainaboard --task kg-link-tail-prediction --dataset fb15k_237 --sub-dataset origin --system-outputs test_distmult.json > log.res
```

```shell
    explainaboard --task kg-link-tail-prediction --custom-dataset-paths ./data/system_outputs/fb15k-237/data_mini.json --system-outputs ./data/system_outputs/fb15k-237/test-kg-prediction-no-user-defined-new.json > report.json
```

## [Aspect-based Sentiment Classification](task_aspect_based_sentiment_classification.md)

Predict the sentiment of a text based on a specific aspect.

**CLI Example**

This is an example with a custom dataset.

```shell
explainaboard --task aspect-based-sentiment-classification --custom-dataset-paths ./data/system_outputs/absa/absa-dataset.txt --system-outputs ./data/system_outputs/absa/absa-example-output.tsv > report.json
```

## [Multiple-choice Cloze]

Fill in a blank based on multiple provided options

**CLI Example**
This is an example using the dataset from `DataLab`

```shell
explainaboard --task cloze-multiple-choice --dataset gaokao2018_np1 --sub-dataset cloze-multiple-choice --metrics CorrectScore --system-outputs ./integration_tests/artifacts/gaokao/rst_2018_quanguojuan1_cloze_choice.json > report.json
```

## [Generative Cloze]

Fill in a blank based on hint

**CLI Example**
This is an example using the dataset from `DataLab`

```shell
explainaboard --task cloze-generative --dataset gaokao2018_np1 --sub-dataset cloze-hint --metrics CorrectScore --system-outputs ./integration_tests/artifacts/gaokao/rst_2018_quanguojuan1_cloze_hint.json > report.json
```

## [Grammatical Error Correction]

Correct errors in a text
**CLI Example**
This is an example using the dataset from `DataLab`

```shell
explainaboard --task grammatical-error-correction --dataset gaokao2018_np1 --sub-dataset writing-grammar --metrics SeqCorrectScore --system-outputs ./integration_tests/artifacts/gaokao/rst_2018_quanguojuan1_gec.json > report.json
```

## Tabular Classification

Classification over tabular data takes in a set of features and predicts a class for
the outputs. The example below is over the `sst2` dataset used in text classification,
but after the text has been vectorized into bag-of-words features. By default the only
features that is analyzed by ExplainaBoard is the `label` feature, so you might want to
specify other features to perform bucketing over using the `metadata` entry in the
dataset `json` file, as is done in `sst2-tabclass-dataset.json` below.

**CLI Examples**

The below example loads a dataset from an existing file:

```shell
explainaboard --task tabular-classification --custom-dataset-paths ./data/system_outputs/sst2_tabclass/sst2-tabclass-dataset.json --system-outputs ./data/system_outputs/sst2/sst2-lstm-output.txt
```

## Tabular Regression

Regression over tabular data is basically the same as tabular classification above, but
the predicted outputs are continuous numbers instead of classes.

**CLI Examples**

The below example loads a dataset from an existing file:

```shell
explainaboard --task tabular-regression --custom-dataset-paths ./data/system_outputs/sst2_tabreg/sst2-tabclass-dataset.json --system-outputs ./data/system_outputs/sst2_tabreg/sst2-tabreg-lstm-output.txt
```

## [Argument Pair Extraction](argument_pair_extraction.md)

This task aim to detect the argument pairs from each passage pair of review and rebuttal.

**CLI Examples**

The below example loads the [`ape`](https://github.com/ExpressAI/DataLab/blob/main/datasets/ape/ape.py) dataset from DataLab:

```shell
explainaboard --task argument-pair-extraction --dataset ape --system-outputs ./data/system_outputs/ape/ape_predictions.txt
```



## [Argument Pair Identification](argument_pair_identification.md)
Given an argument, the task aims to identify one matched argument from a list of arguments.

**CLI Examples**

The example below loads the [`iapi`](https://github.com/ExpressAI/DataLab/blob/main/datasets/iapi/iapi.py) dataset from DataLab: 
```shell
explainaboard --task argument-pair-identification --dataset iapi --system-outputs data/system_outputs/iapi/predictions.txt > report.json
```

## [Meta Evaluation NLG]

Evaluating the reliability of automated metrics for general text generation tasks, such as text summarization.

**CLI Examples**

The below example loads the meval_summeval dataset from DataLab:

```shell
explainaboard --task meta-evaluation-nlg --dataset meval_summeval --sub-dataset coherence --system-outputs ./data/system_outputs/summeval/sumeval_bart.json > report.json
```

## [WMT Metrics Direct Assessment Meta-evaluation](task_meta_evaluation.md)

Evaluating the reliability of automated metrics for [WMT Metrics shared tasks](https://wmt-metrics-task.github.io/)
 using [direct assessment](https://www.statmt.org/wmt16/slides/wmt16-news-da.pdf) (DA).

**CLI Example**

This is an example with a custom dataset.

```shell
explainaboard \
    --task meta-evaluation-wmt-da \
    --custom-dataset-paths ./data/system_outputs/nlg_meta_evaluation/wmt20-DA/cs-en/data.tsv \
    --system-outputs ./data/system_outputs/nlg_meta_evaluation/wmt20-DA/cs-en/score.txt \
    --output-file-type text \
    --output-dir output/cs-en \
    --source-language en \
    --target-language en
```

This is an example with a dataset supported by DataLab, for example
[wmt20_metrics_with_score](https://github.com/ExpressAI/DataLab/blob/main/datasets/wmt20_metrics_with_score/wmt20_metrics_with_score.py).

```shell
explainaboard \
    --task meta-evaluation-wmt-da \
    --dataset wmt20_metrics_with_score \
    --sub-dataset cs-en_1.0.3 \
    --system-outputs ./data/system_outputs/nlg_meta_evaluation/wmt20-DA/cs-en/score_1.0.3.txt \
    --output-file-type text \
    --source-language en \
    --target-language en
```
