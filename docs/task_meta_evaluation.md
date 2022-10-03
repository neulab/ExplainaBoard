# Analyzing Meta Evaluation for NLG Tasks


## Data Preparation

The dataset file format is:
```
SYSName \t SEGID \t TestSet \t src \t ref \t sys \t manualRaw \t manualZ
```
* SYSName is the name of system being scored.
* SEGID is the ID of segment being scored.
* TestSet is the ID of the test set.
* src is the source sentence.
* ref is the reference sentence.
* sys is the sentence to be scored.
* manualRaw is the manual evaluated raw score.
* manualZ is the manual evaluated Z score, standardized for each annotator. 

We have an example dataset file:
* [data.tsv](./data/system_outputs/nlg_meta_evaluation/wmt20-DA/cs-en/data.tsv)

More dataset files can be found at [WMT-DA-20](https://drive.google.com/drive/u/0/folders/1JXpo0yxPLYlNgLbOfP1bzs9z6SOx76Wo).

In order to perform analysis of your results, your system outputs should be a one-column score (the score your metric predicts for the particular example):

```
systemScore1
systemScore2
...
```

We have an example system outputs file:
* [score.txt](./data/system_outputs/nlg_meta_evaluation/wmt20-DA/cs-en/score.txt)




## Performing Basic Analysis
You can load the dataset from an existing file using the
`--custom-dataset-paths` option

```shell
explainaboard \
    --task nlg-meta-evaluation \
    --custom-dataset-paths ./data/system_outputs/nlg_meta_evaluation/wmt20-DA/cs-en/data.tsv \
    --system-outputs ./data/system_outputs/nlg_meta_evaluation/wmt20-DA/cs-en/score.txt \
    --output-file-type text \
    --output-dir output/cs-en \
    --source-language en \
    --target-language en
```



