# Multilingual/Multitask Evaluation

ExplainaBoard currently supports multilingual/multitask evaluation.

## Data Preparation

To achieve multilingual/multitask evaluation, the system outputs files must been organized into JSON
format with following skeleton:

```JSON
{
    "user_defined_metadata_configs": {
        "task_name": "string",
        "system_name": "string",
        "dataset_name": "string",
        "language": "string"
    },
    "predictions": [
        {
            ...
        },
]
}
```

where `...` will be instantiated task-dependently. Specifically,

* [Example](../data/system_outputs/multilingual/json/mlpp/marc/) for text classification
* [Example](../data/system_outputs/multilingual/json/mlpp/xnli/) for sentence pair classification
* [Example](../data/system_outputs/multilingual/json/mlpp/xquad/) for extractive question answering

Once the system outputs are ready, evaluation can be conducted through following steps.

## 1. Generate Analysis Reports for System Output Collections

```shell
explainaboard --system-outputs ./data/system_outputs/multilingual/json/mt5base/xnli/*
```

then all system outputs in the folder `./data/system_outputs/multilingual/json/mt5base/xnli/`

* `test-de_7213.json` (system output on `de` language)
* `test-en_7676.json` (system output on `en` language)
* `test-es_7377.json` (system output on `es` language)
* `test-zh_7117.json` (system output on `zh` language)

 will be processed automatically, and two types of files will be generated:

### (1) analysis reports

* include both overall results and fine-grained analysis of a system
* `json` format
* in the folder `output/reports`

### (2) historgram figures

* each figure could be regarded as a visualization of corresponding analysis report, that provides fine-grained evaluation
   results for a system along certain feature (e.g., sentence length)
* `png` format (would be useful for paper writing)
* in the folder `output/figures`

## 2. More Datasets (tasks), More Systems

We can repeat the above process, using ExplainaBoard SDK to performan analysis for different tasks (datasets) from different languages.
For example,

* Differant languages on `marc` dataset (text classification task) using `mt5base (a.k.a CL-mt5base)` system

```shell
explainaboard --system-outputs ./data/system_outputs/multilingual/json/mt5base/marc/*
```

* Differant languages on `xquad` dataset (text classification task) using `mt5base` system

```shell
explainaboard --system-outputs ./data/system_outputs/multilingual/json/mt5base/xquad/*
```

* Differant languages on `xnli` dataset (natural language inference task) using `mlpp (a.k.a CL-mlpp15out1sum)` system

```shell
explainaboard --system-outputs ./data/system_outputs/multilingual/json/mlpp/xnli/*
```

* Differant languages on `marc` dataset (text classification task) using `mlpp` system

```shell
explainaboard --system-outputs ./data/system_outputs/multilingual/json/mlpp/marc/*
```

* Differant languages on `xquad` dataset (extractive question answwering) using `mlpp` system

```shell
explainaboard --system-outputs ./data/system_outputs/multilingual/json/mlpp/xquad/*
```

## Meta Analysis over Generated Reports

After the above two steps, for all system outputs from

* two systems: `mt5base`, `mlpp`
* three tasks (datasets): `xnli`, `marc`, `xquad`
* multiple languages (e.g., `de`, `en`, `es`, `zh`)

, corresponding reports have been generated. The next step is to perform analysis based on these reports.

ExplainaBoard SDK (CLI) provides some interfaces to achieve this goal, for example

### Print Overall Results of different tasks, models, languages

```shell
explainaboard --reports ./output/reports/*
```

then you will get:

```text
----------------------------------------
Model: CL-mlpp15out1sum, Dataset: xnli
Language:       ar      en      es      zh
Accuracy:       0.696   0.787   0.768   0.731

----------------------------------------
Model: CL-mlpp15out1sum, Dataset: marc
Language:       de      en      es      fr      ja      zh
Accuracy:       0.933   0.915   0.934   0.926   0.915   0.871

----------------------------------------
Model: CL-mlpp15out1sum, Dataset: xquad
Language:       en      es      zh
F1ScoreQA:      0.824   0.782   0.816

----------------------------------------
Model: CL-mt5base, Dataset: xnli
Language:       de      en      es      zh
Accuracy:       0.721   0.768   0.738   0.712

----------------------------------------
Model: CL-mt5base, Dataset: marc
Language:       de      en      es      fr      ja      zh
Accuracy:       0.933   0.920   0.934   0.933   0.914   0.868

----------------------------------------
Model: CL-mt5base, Dataset: xquad
Language:       en      es      zh
F1ScoreQA:      0.812   0.782   0.816
```

### Filter Results based on Customized "Query"

ExplainaBoard also provides interface that users could filter all results with their specified
conditions. For example, if we only care about results on  `xnli` and `marc` datasets

```shell
explainaboard --reports ./output/reports/* --datasets xnli marc
```

Then, following results will be obtained:

```text
----------------------------------------
Model: CL-mlpp15out1sum, Dataset: xnli
Language:       ar      en      es      zh
Accuracy:       0.696   0.787   0.768   0.731

----------------------------------------
Model: CL-mlpp15out1sum, Dataset: marc
Language:       de      en      es      fr      ja      zh
Accuracy:       0.933   0.915   0.934   0.926   0.915   0.871

----------------------------------------
Model: CL-mt5base, Dataset: xnli
Language:       de      en      es      zh
Accuracy:       0.721   0.768   0.738   0.712

----------------------------------------
Model: CL-mt5base, Dataset: marc
Language:       de      en      es      fr      ja      zh
Accuracy:       0.933   0.920   0.934   0.933   0.914   0.868
```

### Aggregated Results based on Customized "Query"

ExplainaBoard SDK allows users to aggregate results along different dimension. For example, if we aim to
know the average performance ove all languages for each dataset,

```shell
explainaboard --reports ./output/reports/* --languages-aggregation average
```

Then following results will be printed:

```text
----------------------------------------
Model: CL-mlpp15out1sum, Dataset: xnli
Language:       all_languages
Accuracy:       0.746

----------------------------------------
Model: CL-mlpp15out1sum, Dataset: marc
Language:       all_languages
Accuracy:       0.916

----------------------------------------
Model: CL-mlpp15out1sum, Dataset: xquad
Language:       all_languages
F1ScoreQA:      0.807

----------------------------------------
Model: CL-mt5base, Dataset: xnli
Language:       all_languages
Accuracy:       0.735

----------------------------------------
Model: CL-mt5base, Dataset: marc
Language:       all_languages
Accuracy:       0.917

----------------------------------------
Model: CL-mt5base, Dataset: xquad
Language:       all_languages
F1ScoreQA:      0.803
```

### System Pair Analysis of Two Groups of Results

Using ExplainaBoard SDK, users can easily get the performance gap (system1 minus system2) over different
datasets (tasks) and different languages.
For example, the following command represent: the performance gap between two systems on all languages
from `mar` and `xnli` datasets.

```shell
explainaboard --reports ./output/reports/* --datasets marc xnli --systems-aggregation minus
```

Then following results will be printed:

```text
----------------------------------------
Model: CL-mlpp15out1sum V.S CL-mt5base, Dataset: marc
Language:       de      zh      fr      ja      es      en
Accuracy:       -0.000  0.003   -0.007  0.001   0.000   -0.005

----------------------------------------
Model: CL-mlpp15out1sum V.S CL-mt5base, Dataset: xnli
Language:       es      en      zh
Accuracy:       0.030   0.020   0.019
```
