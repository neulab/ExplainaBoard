# Examples for CLI

* text-classification:
```shell
explainaboard --task text-classification --system_outputs ./data/system_outputs/sst2/sst2-lstm.tsv
```



* named-entity-recognition:
```shell
  explainaboard --task named-entity-recognition --system_outputs ./data/system_outputs/conll2003/conll2003.elmo
```

* extractive-qa:

```shell
    explainaboard --task extractive-qa --system_outputs ./data/system_outputs/squad/test-qa-squad.json
```


* summarization:
```shell
    explainaboard --task summarization --system_outputs ./data/system_outputs/cnndm/cnndm_mini.bart
```

* text-pair-classification:
```shell
    explainaboard --task text-pair-classification --system_outputs ./data/system_outputs/snli/snli.bert
```

* hellaswag

```shell
    explainaboard --task hellaswag --system_outputs ./data/system_outputs/hellaswag/hellaswag.random
```
