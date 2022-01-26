# Support New Features

Take `text_classification` task for example, supposing we aim to add
 a new feature `token_number` to bucket test set for fine-grained evaluation.

## Feature Declaration

We need to declare the feature in the task processor module, for example:
`explainaboard/processors/text_classification.py`

```python
class TextClassificationProcessor(Processor):
    _task_type = TaskType.text_classification
    _features = feature.Features({
        ...
        ...
        "token_number": feature.Value(dtype="float",
                                      is_bucket=True,
                                      bucket_info=feature.BucketInfo(
                                          _method="bucket_attribute_specified_bucket_value",
                                          _number=4,
                                          _setting=())),

    })
```
where
* `dtype` represents the data type of the feature
    * `float` for continuous feature
    * `string` for discrete feature
* `is_bucket`: whether this feature will be used for bucketing test set
    * True
    * False
* `bucekt_info`: the specific information for the bucketing operation
    * `method`: bucketing methods
        * `bucket_attribute_specified_bucket_value`: when `dtype = "float"`
        * `bucket_attribute_discrete_value`: when `dtype = "string"`
    * `number`: the number of buckets
    * `_setting`: the hyperparameter of bucketing interval
        * `_setting=()` when `dtype = "float"`
         * `_setting=1` when `dtype = "string"`   
    

## Feature Implementation and Calculation

We need to define the feature function and apply it to each sample
of the dataset in the module: `explainaboard/builders/text_classification`

There are flexible way to achieve this but remember the final goal is:

(1) given raw features stored in the dictionary `dict_sysout`, for example,
* `dict_sysout['text']`: the raw input text
* `dict_sysout['label']`: the gold label of the input text
(2) how to calculate the new feature and add it into `dict_sysout`, here
  is `dict_sysout[token_number]`
  
Following highlights the core implementation inside the `class TCExplainaboardBuilder` in `explainaboard/builders/summarization`
```python
class TCExplainaboardBuilder:
    ...
    def _get_token_number(self, text):
        return len(text)

    def _complete_feature(self):
 
 
        bucket_features = self._info.features.get_bucket_features()
        for _id, dict_sysout in enumerate(self._system_output):
            # Get values of bucketing features
            for bucket_feature in bucket_features:
                if bucket_feature == "token_number":
                    feature_value = self._get_token_number(dict_sysout["text"])
                    dict_sysout[bucket_feature] = feature_value
            self._data[_id] = dict_sysout
            yield _id, dict_sysout


```