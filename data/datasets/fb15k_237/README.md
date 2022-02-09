# FB15K_237


## How to load the dataset using `DataLab`?

```python
# pip install --upgrade pip
# pip install datalabs
from datalabs import load_dataset
dataset = load_dataset("fb15k_237",'readable') # you can print one example by: print(dataset['test'][0])
```

Then will get
```
DatasetDict({
    train: Dataset({
        features: ['head', 'link', 'tail'],
        num_rows: 272115
    })
    validation: Dataset({
        features: ['head', 'link', 'tail'],
        num_rows: 17535
    })
    test: Dataset({
        features: ['head', 'link', 'tail'],
        num_rows: 20466
    })
})
```

