# CLI Examples

Below is a list of examples of how to analyze different tasks.
Click the link on the task name for more details, or when no link exists you can open the example data to see what the file format looks like.

We welcome contributions of [more tasks](add_new_tasks.md), or detailed documentation for tasks where the documentation does not yet exists! Please open an issue or file a PR.

* [text-classification](task_text):
```shell
explainaboard --task text-classification --system_outputs ./data/system_outputs/sst2/sst2-lstm.tsv
```

* named-entity-recognition:
```shell
explainaboard --task named-entity-recognition --system_outputs ./data/system_outputs/conll2003/conll2003.elmo
```

* extractive-qa:

```shell
explainaboard --task extractive-qa --system_outputs ./data/system_outputs/squad/testset-en.json
```


* summarization:
```shell
explainaboard --task summarization --system_outputs ./data/system_outputs/cnndm/cnndm_mini.bart
```

* [text-pair-classification](task_text_pair_classification.md):
```shell
explainaboard --task text-pair-classification --system_outputs ./data/system_outputs/snli/snli.bert
```

* [hellaswag](task_hellaswag.md)

```shell
explainaboard --task hellaswag --system_outputs ./data/system_outputs/hellaswag/hellaswag.random
```
