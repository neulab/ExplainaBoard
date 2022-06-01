# Analyzing Text Regression


## Data Preparation

The dataset file fomat is:
```
SYSName \t SEGID \t TestSet \t src \t ref \t sys \t manualRaw \t manualZ
```

We have an example dataset file:
* [data.tsv](./data/system_outputs/text_regression/wmt20-DA/cs-en/data.tsv)

More dataset files can be found at [WMT-DA-20](https://drive.google.com/drive/u/0/folders/1JXpo0yxPLYlNgLbOfP1bzs9z6SOx76Wo).

In order to perform analysis of your results, your system outputs should add one column of system score at the base of dataset file:

```
SYSName \t SEGID \t TestSet \t src \t ref \t sys \t manualRaw \t manualZ \t systemScore
```

We have an example system outputs file:
* [score.tsv](./data/system_outputs/text_regression/wmt20-DA/cs-en/score.tsv)




## Performing Basic Analysis
You can load the dataset from an existing file using the
`--custom_dataset_paths` option

```shell
explainaboard   --task text-regression --custom_dataset_paths ./data/system_outputs/text_regression/wmt20-DA/cs-en/data.tsv --system_outputs ./data/system_outputs/text_regression/wmt20-DA/cs-en/score.tsv --output_file_type tsv --output_dir output/cs-en --source_language en --target_language en --metrics SysPearsonCorr SegKtauCorr RootMeanSquareError
```



