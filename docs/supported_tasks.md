# Currently Supported Tasks

Below is a (somewhat incomplete) list of tasks that ExplainaBoard currently supports, along with examples of how to analyze different tasks.
Click the link on the task name for more details, or when no link exists you can open the example data to see what the file format looks like.

All of the examples below will output a json report to standard out, and you can use Python's pretty printing tool to see it in a more readable format (add `| python -m json.tool` at the end of any of the lines).

We welcome contributions of [more tasks](add_new_tasks.md), or detailed documentation for tasks where the documentation does not yet exist! Please open an issue or file a PR.

**Table of Contents**
* [Text Classification](#text-classification)
* [Text Pair Classification](#text-pair-classification)
* [Summarization](#summarization)
* [Named Entity Recognition](#named-entity-recognition)
* [Extractive QA](#extractive-qa)
* [Hellaswag](#hellaswag)


## [Text Classification](task_text_classification.md)

Classification of text into different categories.

**CLI Example**
```shell
explainaboard --task text-classification --system_outputs ./data/system_outputs/sst2/sst2-lstm.tsv
```

**Class**
* `TaskType.text_classification`

**Supported Formats**
* `FileType.tsv`
  
**Supported Metrics**
* `F1score`
* `Accuracy`


## [Text Pair Classification](task_text_pair_classification.md)

Classification of pairs of text, such as natural language inference.

**CLI Example**
```shell
explainaboard --task text-pair-classification --system_outputs ./data/system_outputs/snli/snli.bert
```

## [Summarization](task_summarization.md)

Summarization of longer texts into shorter texts.

**CLI Example**
```shell
explainaboard --task summarization --system_outputs ./data/system_outputs/cnndm/cnndm_mini.bart --metrics bart_score_summ rouge2
```
where different metrics should be separated by space

**Class**
* `TaskType.summarization`
  
**Supported Formats**
* `FileType.tsv`
  
**Supported Metrics**
* `bart_score_cnn_hypo_ref`
* `bart_score_summ`
* `bart_score_mt`
* `bert_score_p`
* `bert_score_r`
* `bert_score_f`
* `bleu`
* `chrf`
* `comet`
* `comet_qe`
* `mover_score`
* `prism`
* `prism_qe`
* `rouge1`
* `rouge2`
* `rougeL`





## Named Entity Recognition

Recognizing the entities in text.

**CLI Example**
```shell
explainaboard --task named-entity-recognition --system_outputs ./data/system_outputs/conll2003/conll2003.elmo
```

**Class**
* `TaskType.named_entity_recognition`

**Supported Format**
* `FileType.conll`
  
**Supported Metrics**
* `f1_score_seqeval`
 

## [Extractive QA](task_extractive_qa.md)

Extractive QA tasks, such as SQuAD.

**CLI Example**
```shell
explainaboard --task extractive-qa --system_outputs ./data/system_outputs/squad/test-qa-squad.json
```

**Class**
* `TaskType.extractive_qa`

**Supported Format**
* `FileType.json` (same format with squad)
  
**Supported Metric**
* `f1_score_qa`
* `exact_match_qa`
 

## [Hellaswag](task_hellaswag.md)

**CLI Example**
```shell
explainaboard --task hellaswag --system_outputs ./data/system_outputs/hellaswag/hellaswag.random
```
