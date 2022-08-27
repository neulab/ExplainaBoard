# Analyzing Extractive QA Task


In this file we describe how to analyze models trained on extractive QA datasets, for example
[`squad`](http://datalab.nlpedia.ai/#/normal_dataset/6163a29beb9872f33252b01b/dataset_samples).


## Data Preparation

In order to perform analysis of your results, they should be in the following
JSON format:

```json
{
        "predicted_answers": {
            "text": "Kawann Short"
        }
    }
```
where 
* `predicted_answers`: denotes the predicted answers

An example system output file is here:
* [squad_mini-example-output.json](https://github.com/neulab/ExplainaBoard/blob/main/data/system_outputs/squad/squad_mini-example-output.json) 

## Performing Basic Analysis

The below example loads the `squad` dataset from DataLab. There is an [open issue](https://github.com/neulab/ExplainaBoard/issues/239) that prevents the specification of a dataset split, so this will not work at the moment. But we are working on it.
```shell
explainaboard --task qa-extractive --dataset squad --system-outputs MY_FILE > report.json
```
* `--task`: denotes the task name.
* `--system-outputs`: denote the path of system outputs. Multiple one should be
  separated by space, for example, system1 system2
* `--dataset`:optional, denotes the dataset name
* `report.json`: the generated analysis file with json format. . Tips: use a json viewer
  like [this one](http://jsonviewer.stack.hu/) for better interpretation.

You can use a custom dataset directly:
```shell
explainaboard --task qa-extractive --custom-dataset-paths ./data/system_outputs/squad/squad_mini-dataset.json --system-outputs ./data/system_outputs/squad/squad_mini-example-output.json > report.json
```

The dataset can be in the following format:
```json
{
  "id": "4",
  "context": "The Panthers defense gave up just 308 points, ranking sixth in the league, while also leading the NFL in interceptions with 24 and boasting four Pro Bowl selections. Pro Bowl defensive tackle Kawann Short led the team in sacks with 11, while also forcing three fumbles and recovering two. Fellow lineman Mario Addison added 6½ sacks. The Panthers line also featured veteran defensive end Jared Allen, a 5-time pro bowler who was the NFL's active career sack leader with 136, along with defensive end Kony Ealy, who had 5 sacks in just 9 starts. Behind them, two of the Panthers three starting linebackers were also selected to pl",
  "question": "How many balls did Josh Norman intercept?",
  "answers":
  {
    "answer_start": 192,
    "text": "Kawann Short"
  }
,
  "predicted_answers": {
    "text": "Kawann Short"
  }
}
```

If there are multiple true answers, the above format cam be modified to:
```json
{
        "id": "4",
        "context": "The Panthers defense gave up just 308 points, ranking sixth in the league, while also leading the NFL in interceptions with 24 and boasting four Pro Bowl selections. Pro Bowl defensive tackle Kawann Short led the team in sacks with 11, while also forcing three fumbles and recovering two. Fellow lineman Mario Addison added 6½ sacks. The Panthers line also featured veteran defensive end Jared Allen, a 5-time pro bowler who was the NFL's active career sack leader with 136, along with defensive end Kony Ealy, who had 5 sacks in just 9 starts. Behind them, two of the Panthers three starting linebackers were also selected to pl",
        "question": "How many balls did Josh Norman intercept?",
        "answers":
          {
            "answer_start": [192,200],
            "text": "[Kawann Short, Peter]"
          }
        ,
        "predicted_answers": {
            "text": "Kawann Short"
        }
    }
```
