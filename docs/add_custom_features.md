# Add custom features for custom analysis

If you want to perform custom analysis with your custom features that are not supported in the original task processors, and these features are only related to one particular system/dataset instead of the task itself, you can define the custom features and analysis in the `metadata` section in the output JSON file. 

## Example of bucket analysis with discrete feature
Here is the output json format for bucket analysis with custom discrete features with `"dtype": "string"`, such as the subject of the sentence, etc.
```
{
  "metadata": {
    "custom_features": {
      "feature-level": {
        "custom-feature-name": {
          "dtype": "string",
          "description": "(optional) description of the feature"
        }
      }
    },
    "custom_analyses": [
      {
        "cls_name": "BucketAnalysis",
        "level": "feature-level",
        "feature": "custom-feature-name",
        "num_buckets": 15,
        "method": "discrete"
        "sample_limit": 50
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
* `feature-level` represents the fine-grained level of the analysis
    * `example` for sentence-level analysis
    * `span` for span-level analysis (e.g. in named entity recognition analysis)
    * `token` for token-level analysis (e.g. in named entity conditional text generation)
* `num_bucket`: the number of buckets to be used
* `sample_limit`: if the number of examples are larger than sample_limit, randomly select sample_limit examples for analysis

## Example of bucket analysis with continuous feature
Here is the output json format for bucket analysis with custom continuous features with `"dtype": "float"`, such as the count of particular words, the output logits/probability, etc.
```
{
  "metadata": {
    "custom_features": {
      "feature-level": {
        "custom-feature-name": {
          "dtype": "float",
          "description": "(optional) description of the feature"
        }
      }
    },
    "custom_analyses": [
      {
        "cls_name": "BucketAnalysis",
        "level": "feature-level",
        "feature": "custom-feature-name",
        "num_buckets": 15,
        "method": "continuous"
        "sample_limit": 50
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
* `feature-level` represents the fine-grained level of the analysis
    * `example` for sentence-level analysis
    * `span` for span-level analysis (e.g. in named entity recognition analysis)
    * `token` for token-level analysis (e.g. in named entity conditional text generation)
* `num_bucket`: the number of buckets to be used
* `sample_limit`: if the number of examples are larger than sample_limit, randomly select sample_limit examples for analysis

## Example of ComboCountAnalysis
ComboCountAnalysis is used to count feature combinations (e.g. for confusion matrices). It will
return counts of each combination of values for the features named in `features`.
```
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
* `feature-level` represents the fine-grained level of the analysis
    * `example` for sentence-level analysis
    * `span` for span-level analysis (e.g. in named entity recognition analysis)
    * `token` for token-level analysis (e.g. in named entity conditional text generation)
* `features` should be a list of feature names where each feature are predefined. These features can be true_label, predicted_label, or other custom features.

## Example output json files
* [text-classification](https://github.com/neulab/ExplainaBoard/blob/main/data/system_outputs/text-classification-custom-feature-example.json)
* [machine-translation](https://github.com/neulab/ExplainaBoard/blob/main/integration_tests/artifacts/machine_translation/output_with_features.json)

## Note 
When running analysis with SDK command, `--custom-dataset-file-type` and `--output-file-type json` are required. 

Example command:
```
explainaboard --task text-classification --custom-dataset-paths path_to_my_custom_data.tsv --system-outputs path_to_my_output_with_custom_features.json --report-json path_to_my_report.json --custom-dataset-file-type tsv --output-file-type json
```