# Support New Features

Before you read this, it might be a good idea to read the doc on how to
[add new tasks](add_new_tasks.md) to get an idea of the full structure of how
tasks are implemented.

Take the `text_classification` task for example, suppose that we aim to add
 a new feature `token_number` to bucket the test set for fine-grained evaluation.

## Adding Features by Directly Modifying the Processor

If we want to declare a new feature we can do so by directly modifying the processor
module corresponding to its task, in this case:
`explainaboard/processors/text_classification.py`

To add a feature that is analyzed for each example:
1. add an entry or two to the `features` list
2. make sure that a `BucketAnalysis` object is added to the `analyses` list. For many
   processors, float features are added automatically, but string features need to be
   added explicitly, as below.

```python
from explainaboard.processors.processor import Processor
from explainaboard.analysis.analyses import AnalysisLevel, BucketAnalysis
from explainaboard.analysis.feature import FeatureType
from explainaboard.analysis import feature
from explainaboard.analysis.feature_funcs import count_tokens
from explainaboard.utils.typing_utils import unwrap

class TextClassificationProcessor(Processor):
    ...

    def default_analyses(self) -> list[AnalysisLevel]:
        # ---- Features ----
        features: dict[str, FeatureType] = {
            "chars_per_word": feature.Value(
                dtype="float",
                description="character count divided by token count",
                func=lambda info, x, c: float(len(x['text'])) / count_tokens(info, x['text']),
            ),
            "contains_question": feature.Value(
                dtype="string",
                description="",
                func=lambda info, x, c: "yes" if "?" in x['text'] else "no",
            ),
            # ...
        }
        # ---- Analyses ----
        continuous_features = [
            k for k, v in features.items() if ('float' in unwrap(v.dtype))
        ]
        analyses: list[BucketAnalysis] = [
            BucketAnalysis(
                feature="contains_question",
                method="discrete",
                number=2,
            )
        ] + [BucketAnalysis(x, method="continuous") for x in continuous_features]
        # ...


```
where
* `dtype` represents the data type of the feature
    * `float` for continuous feature
    * `string` for discrete feature
* `description`: the description of the feature
* `func`: a function to calculate the feature, with three arguments
   * `info`: the SysOutputInfo object
   * `x`: the original example data from the system output/dataset
   * `c`: the `AnalysisCase` corresponding to this example

Note that it is possible to define features not only over the whole example, but also
over individual tokens or spans. You can take a look at `sequence_labeling.py` or 
`conditional_generation.py` for examples of this.

## Features and Unittests

Note that you may need to change the test cases in the module relevant to your task
in `integration_tests/` upon adding a new feature, such as changing the number of
features in asserts to match the current number of features or testing that feature
independently.

## Training-set Dependent Features

If you want to add features that are dependent on the training set, you will need to

1. Implement a `get_statistics` function that saves the statistics from the training set
2. Declare `require_training_set=True` in the feature definition
3. Use the passed-in `statistics` object to access the training set statistics when calculating features

It is probably best to learn by example, so you can take a look at `get_num_oov` in
`processors/text_classification.py`.
