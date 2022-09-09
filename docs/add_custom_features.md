# Add custom features for custom analysis

If you want to perform custom analysis with your custom features that are not supported in the original task processors, and these features are only related to one particular system/dataset, you can define the custom features and analysis in the `metadata` section in the output JSON file. 

Here is the template of the output JSON file with custom metadata.
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
        "method": "discrete/continuous"
      }
    ]
  },
  "examples": [
    {
      "output-json-key": "model-output",
      "custom-feature-name": "feature-value"
    }
  ]
}
```
where
* `feature-level` represents the fine-grained level of the analysis
    * `example` for sentence-level analysis
    * `span` for span-level analysis
* `dtype` represents the data type of the feature
    * `float` for continuous feature
    * `string` for discrete feature
* `cls_name` means the analysis type
    * `BucketAnalysis` requires the following fields to be defined
    ```
    BucketAnalysis(
            description=dikt.get('description'),
            level=dikt['level'],
            feature=dikt['feature'],
            method=dikt.get('method', 'continuous'),
            number=dikt.get('number', 4),
            setting=dikt.get('setting'),
            sample_limit=dikt.get('sample_limit', 50),
        )
    ```
    * `ComboCountAnalysis` requires the following fields to be defined
    ```
    ComboCountAnalysis(
            description=dikt.get('description'),
            level=dikt['level'],
            features=dikt['features'],
        )
    ```

Example output files with custom features:
* [text-classification](https://github.com/neulab/ExplainaBoard/blob/main/data/system_outputs/text-classification-custom-feature-example.json)
* [machine-translation](https://github.com/neulab/ExplainaBoard/blob/main/integration_tests/artifacts/machine_translation/output_with_features.json)
