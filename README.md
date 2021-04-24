# ExplainaBoard: An Explainable Leaderboard for NLP

[**Introduction**](##introduction) | 
[**Website**](#website) |
[**Download**](#download-system-output) |
[**Backend**](#test-your-results) |
[**Paper**](https://arxiv.org/pdf/2104.06387.pdf) |
[**Video**](https://www.youtube.com/watch?v=3X6NgpbN_GU) |
[**Bib**](http://explainaboard.nlpedia.ai/explainaboard.bib)


<img src="./fig/logo-full-v2.png" width="800" class="center">

## Introduction
### ExplainaBoard is an interpretable, interactive and reliable leaderboard with seven (so far) new features (F) compared with generic leaderboard.
* F1: *Single-system Analysis*: What is a system good or bad at?
* F2: *Pairwise Analysis*: Where is one system better (worse) than another?
* F3: *Data Bias Analysis*: What are the characteristics of different evaluated datasets?
* F5: *Common errors*: What are common mistakes that top-5 systems made?
* F6: *Fine-grained errors*: where will errors occur?
* F7: *System Combination*: Is there potential complementarity between different systems?


<img src="./fig/intro.png" width="400" class="center">

<img src="./fig/demo.gif" width="800" class="center">




## Website
We deploy ExplainaBoard as a [Web toolkit](http://explainaboard.nlpedia.ai/), which includes 9 NLP tasks, 
40 datasets and 300 systems. 

## Task 

| Task                     | Sub-task         | Dataset | Model | Attribute | 
|--------------------------|------------------|---------|-------|-----------|  
|				           | Sentiment		  | 8       | 40    | 2         |
| Text Classification      | Topics           | 4       | 18    | 2         |
|					       | Intention        | 1       | 3     | 2         |
| Text-Span Classification | Aspect Sentiment | 4       | 20    | 4         |
| Text pair Classification | NLI              | 2       | 6     | 7         |
|                          | NER              | 3       | 74    | 9         |
| Sequence Labeling	       | POS              | 3       | 14    | 4         |	
| 					       | Chunking         | 3       | 14    | 9         |
| 					       | CWS              | 7       | 64    | 7         |
| Structure Prediction     | Semantic Parsing | 4       | 12    | 4         | 
| Text Generation          | Summarization    | 2       | 36    | 7         | 


 ## Install Dependency Libararies （python 3.X）
```
pip install -r requirements.txt
```

## Description of Each Directory
* `task-[task_name]`: fine-grained analysis for each task, 
  aiming to generating fine-grained analysis results with the json format.
  For example, task-mlqa can calculate the fine-graied F1 scores for different systems,
  and output corresponding json files in task-mlqa/output/ .
  
* `meta-eval` is a sort of controller, which can be used to start the fine-graind anlsysis of all
tasks, and analyze output json files.

    - calculate fine-grained results for all tasks: ./meta-eval/run-allTasks.sh
    ```js
        cd ./meta-eval/
        ./run-allTasks.sh
     ```
  
    - merge json files of all tasks into a csv file, which would be useful for further SQL import:
    ./meta-eval/genCSV/json2csv.py
  
    ```js
        cd ./meta-eval/genCSV/json2csv.py
        python json2csv.py > xtreme.csv
    ```

* `src` stores some auxiliary codes.
