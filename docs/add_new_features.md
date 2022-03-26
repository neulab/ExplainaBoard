# Support New Features

Before oyu read this, it might be a good idea to read the doc on how to
[add new tasks](add_new_tasks.md) to get an idea of the full structure of how
tasks are implemented.

Take the `text_classification` task for example, suppose that we aim to add
 a new feature `token_number` to bucket the test set for fine-grained evaluation.

## Feature Declaration

We need to declare the new feature `token_number` in the task processor module
corresponding to its task, in this case: `explainaboard/processors/text_classification.py`

```python
class TextClassificationProcessor(Processor):
    @classmethod
    def task_type(cls) -> TaskType:
        return TaskType.text_classification

    @classmethod
    def features(cls) -> feature.Features:
        return feature.Features({
            ...
            ...
            "token_number": feature.Value(dtype="float",
                                        is_bucket=True,
                                        bucket_info=feature.BucketInfo(
                                            _method="bucket_attribute_specified_bucket_value",
                                            _number=4,
                                            _setting=())),

        })

    @classmethod
    def default_metrics(cls) -> List[str]:
        return ["Accuracy"]
```
where
* `dtype` represents the data type of the feature
    * `float` for continuous feature
    * `string` for discrete feature
* `is_bucket`: whether this feature will be used for bucketing test set
    * True
    * False
* `bucket_info`: the specific information for the bucketing operation
    * `method`: bucketing methods
        * `bucket_attribute_specified_bucket_value`: when `dtype = "float"`
        * `bucket_attribute_discrete_value`: when `dtype = "string"`
    * `number`: the number of buckets
    * `_setting`: the hyperparameter of bucketing interval
        * `_setting=()` when `dtype = "float"`
         * `_setting=1` when `dtype = "string"`   
    

## Feature Implementation and Calculation

After declaring the new feature `token_number`, we need to define a corresponding
feature function `_get_token_number(self, text)` and apply it to each sample
of the dataset in the module: `explainaboard/processors/text_classification.py`

There are a couple of flexible ways to achieve this but remember the final goal is:

(1) given raw features stored in the dictionary `existing_features`, for example,
* `existing_features['text']`: the raw input text
* `existing_features['label']`: the gold label of the input text

(2) how to calculate the new feature and add it into `existing_features`, which
  is `existing_features[token_number]` in the current context
  
The following highlights the core implementation inside `class TCExplainaboardBuilder` in `explainaboard/builders/summarization`
```python
class TCExplainaboardProcessor:
    ...
    def _get_sentence_length(self, existing_features: dict):
        return len(self._tokenizer(existing_features["text"]))
```

## Features and Unittests

Note that you may need to change the test cases in the module relevant to your task
in `explainaboard/tests/` upon adding a new feature, such as changing the number of
features in asserts to match the current number of features or testing that feature
independently.

## Training-set Dependent Features

If you want to add features that are dependent on the training set, you will need to

1. Implement a `get_statistics` function that saves the statistics from the training set
2. Declare `require_training_set=True` in the feature definition
3. Use the passed-in `statistics` object to access the training set statistics when calculating features

It is probably best to learn by example, so you can take a look at `get_num_oov` in
`processors/text_classification.py`.
