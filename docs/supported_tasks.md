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
* [Extractive QA_SQuAD](#extractive-qa-squad)
* [Hellaswag](#hellaswag)
* [KG-Link-Tail-Prediction](#kg-link-tail-prediction)
* [Aspect-based Sentiment Classification](#aspect-based-sentiment-classification)






## [KG-Link-Tail-Prediction](task_kg_link_tail_prediction.md)
Predicting the tail entity of missing links in knowledge graphs

**CLI Example**
```shell
    explainaboard --task kg-link-tail-prediction --system_outputs ./data/system_outputs/fb15k-237/test-kg-link-tail-prediction.json > report.json
    # or
    explainaboard --task kg-link-tail-prediction --system_outputs ./data/system_outputs/fb15k-237/test-kg-link-tail-prediction.json --dataset fb15k_237 > report.json
```



**Supported Formats**
* `FileType.json`
  
**Supported Metrics**
* `Hits`




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
* `bart_score_cnn_hypo_ref`: [BARTScore](https://arxiv.org/abs/2106.11520) is a sequence to sequence framework based on pre-trained language model BART.  `bart_score_cnn_hypo_ref` uses the CNNDM finetuned BART. It calculates the average generation score of `Score(hypothesis|reference)` and `Score(reference|hypothesis)`.
* `bart_score_summ`: [BARTScore](https://arxiv.org/abs/2106.11520) using the CNNDM finetuned BART. It calculates `Score(hypothesis|source)`.
* `bart_score_mt`: [BARTScore](https://arxiv.org/abs/2106.11520) using the Parabank2 finetuned BART. It calculates the average generation score of `Score(hypothesis|reference)` and `Score(reference|hypothesis)`.
* `bert_score_p`: [BERTScore](https://arxiv.org/abs/1904.09675) is a metric designed for evaluating translated text using BERT-based matching framework. `bert_score_p` calculates the BERTScore precision.
* `bert_score_r`: [BERTScore](https://arxiv.org/abs/1904.09675) recall.
* `bert_score_f`: [BERTScore](https://arxiv.org/abs/1904.09675) f score.
* `bleu`: [BLEU](https://aclanthology.org/P02-1040.pdf) measures modified ngram matches between each candidate translation and the reference translations. 
* `chrf`: [CHRF](https://aclanthology.org/W15-3049/) measures the character-level ngram matches between hypothesis and reference.
* `comet`: [COMET](https://aclanthology.org/2020.emnlp-main.213/) is a neural framework for training multilingual machine translation evaluation models. `comet` uses the `wmt20-comet-da` checkpoint which utilizes source, hypothesis and reference.
* `comet_qe`: [COMET](https://aclanthology.org/2020.emnlp-main.213/) for quality estimation. `comet_qe` uses the `wmt20-comet-qe-da` checkpoint which utilizes only source and hypothesis.
* `mover_score`: [MoverScore](https://arxiv.org/abs/1909.02622) is a metric similar to BERTScore. Different from BERTScore, it uses the Earth Mover’s Distance instead of the Euclidean Distance.
* `prism`: [PRISM](https://arxiv.org/abs/2004.14564) is a sequence to sequence framework trained from scratch. `prism` calculates the average generation score of `Score(hypothesis|reference)` and `Score(reference|hypothesis)`.
* `prism_qe`: [PRISM](https://arxiv.org/abs/2004.14564) for quality estimation. It calculates `Score(hypothesis| source)`.
* `rouge1`: [ROUGE-1](https://aclanthology.org/W04-1013/) refers to the overlap of unigram (each word) between the system and reference summaries.
* `rouge2`: [ROUGE-2](https://aclanthology.org/W04-1013/) refers to the overlap of bigrams between the system and reference summaries.
* `rougeL`: [ROUGE-L](https://aclanthology.org/W04-1013/) refers to the longest common subsequence between the system and reference summaries.





## Named Entity Recognition

Recognizing the entities in text.

**CLI Example**
```shell
explainaboard --task named-entity-recognition --system_outputs ./data/system_outputs/conll2003/conll2003.elmo

or 

explainaboard --task named-entity-recognition --dataset conll2003 --system_outputs ./data/system_outputs/conll2003/conll2003.elmo

```

**Class**
* `TaskType.named_entity_recognition`

**Supported Format**
* `FileType.conll`
  
**Supported Metrics**
* `f1_score_seqeval`
 

## [Extractive QA_SQuAD](task_extractive_qa_squad.md)

Extractive QA tasks, such as SQuAD.

**CLI Example**
```shell
explainaboard --task extractive-qa-squad --system_outputs ./data/system_outputs/squad/test-qa-squad.json
```

**Class**
* `TaskType.extractive_qa_squad`

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

## [Aspect-based Sentiment Classification](task_aspect_based_sentiment_classification.md)
Predict the sentiment of a text based on a specific aspect.

**CLI Example**
```shell
explainaboard --task aspect-based-sentiment-classification --system_outputs ./data/system_outputs/absa/test-aspect.tsv > ./data/reports/report_absa.json
```

**Class**
* `TaskType.aspect_based_sentiment_classification`

**Supported Formats**
* `FileType.tsv`
  
**Supported Metrics**
* `F1score`
* `Accuracy`