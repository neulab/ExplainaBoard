# Add custom features for custom analysis

If you want to perform custom analysis with your custom features that are not supported
in the original task processors, and these features are only related to one particular
system/dataset instead of the task itself, you can define the custom features and
analysis in the `metadata` section in the output JSON file.

## Example of bucket analysis

Here is the output json format for bucket analysis with custom features. Discrete
features should have `"dtype": "string"`, such as the subject of the sentence, etc.
Continuous features should have `"dtype": "float"`, such as the count of particular
words, the output logits/probability, etc.

```json
{
  "metadata": {
    "custom_features": {
      "feature-level": {
        "discrete-custom-feature-name": {
          "cls_name": "Value",
          "dtype": "string",
          "description": "(optional) description of the feature"
        },
        "continuous-custom-feature-name": {
          "cls_name": "Value",
          "dtype": "float",
          "description": "(optional) description of the feature"
        }
      }
    },
    "custom_analyses": [
      {
        "cls_name": "BucketAnalysis",
        "level": "feature-level",
        "feature": "discrete-custom-feature-name",
        "num_buckets": 15,
        "method": "discrete",
        "sample_limit": 50
      },
      {
        "cls_name": "BucketAnalysis",
        "level": "feature-level",
        "feature": "continuous-custom-feature-name",
        "num_buckets": 15,
        "method": "continuous",
        "sample_limit": 50
      }
    ]
  },
  "examples": [
    {
      "predicted_label": "true",
      "discrete-custom-feature-name": "discrete-feature-value",
      "continuous-custom-feature-name": "continuous-feature-value"
    }
  ]
}
```

where

* `feature-level`: the fine-grained level of the analysis
  * `example` for sentence-level analysis
  * `span` for span-level analysis (e.g. in named entity recognition analysis)
  * `token` for token-level analysis (e.g. in named entity conditional text generation)
* `num_bucket`: the number of buckets to be used
* `sample_limit`: if the number of examples are larger than sample_limit, randomly
  select sample_limit examples for analysis

## Example of ComboCountAnalysis

ComboCountAnalysis is used to count feature combinations (e.g. for confusion matrices).
It will return counts of each combination of values for the features named in `features`.

```json
{
  "metadata": {
    "custom_analyses": [
      {
        "cls_name": "ComboCountAnalysis",
        "level": "feature-level",
        "description": "(optional) description of the analysis",
        "features": ["feature-name-1", "feature-name-2"]
      }
    ]
  },
  "examples": [
    {
      "predicted_label": "true",
      "custom-feature-name": "feature-value"
    }
  ]
}
```

where

* `feature-level`: the fine-grained level of the analysis
  * `example` for sentence-level analysis
  * `span` for span-level analysis (e.g. in named entity recognition analysis)
  * `token` for token-level analysis (e.g. in named entity conditional text generation)
* `features`: a list of feature names where each feature are predefined. These
  features can be true_label, predicted_label, or other custom features.

## Example of CalibrationAnalysis

CalibrationAnalysis measures [calibration](https://arxiv.org/abs/1706.04599),
the association between a model-predicted probability and the likelihood
of the answer being correct. It must be used on tasks that use the
Accuracy metric. For tasks that use the Accuracy metric, by default,
calibration analysis is automatically performed if your model inputs
have the `confidence` feature. You can directly add the `confidence`
values to the examples in a json-formatted system output file.
You may also customize calibration analysis with custom feature name
and settings.

Default calibration analysis format [example](../data/system_outputs/absa/absa-example-output-confidence.json):

```json
{
  "examples": [
    {
      "predicted_label": "positive",
      "confidence": 0.22101897026820283
    }
  ]
}
```

Custom calibration analysis format [example](../data/system_outputs/absa/absa-example-output-custom-calibration-analysis.json):

```json
{
  "metadata": {
    "custom_features": {
      "example": {
        "probability": {
          "cls_name": "Value",
          "dtype": "float",
          "description": "model-predicted probability"
        }
      }
    },
    "custom_analyses": [
      {
        "cls_name": "CalibrationAnalysis",
        "level": "example",
        "feature": "probability",
        "num_buckets": 5,
        "sample_limit": 50
      }
    ]
  },
  "examples": [
    {
      "predicted_label": "positive",
      "probability": 0.22101897026820283
    }
  ]
}
```

where

* `feature`: the customized name of the confidence feature
* `num_bucket`: the number of buckets of the same size between interval [0, 1]

## Example output json files

* [text-classification](https://github.com/neulab/ExplainaBoard/blob/main/data/system_outputs/text-classification-custom-feature-example.json)
* [machine-translation](https://github.com/neulab/ExplainaBoard/blob/main/integration_tests/artifacts/machine_translation/output_with_features.json)

## Note

When running analysis with SDK command, `--custom-dataset-file-type` and
`--output-file-type json` are required.

Example command:

```shell
explainaboard --task text-classification --custom-dataset-paths path_to_my_custom_data.tsv --system-outputs path_to_my_output_with_custom_features.json --report-json path_to_my_report.json --custom-dataset-file-type tsv --output-file-type json
```
