# ExplainaBoard: An Explainable Leaderboard for NLP

[**Introduction**](##introduction) | 
[**Website**](#website) |
[**Download**](#download-system-outputs) |
[**Backend**](#test-your-results) |
[**Paper**](https://arxiv.org/pdf/2104.06387.pdf) |
[**Video**](https://www.youtube.com/watch?v=3X6NgpbN_GU) |
[**Bib**](http://explainaboard.nlpedia.ai/explainaboard.bib)


<img src="./fig/logo-full-v2.png" width="800" class="center">



## Test Your Results
```
pip install -r requirements.txt
python setup.py install
```

### Example

```
interpret-eval -h
interpret-eval --task chunk --systems ./interpret_eval/example/test-conll00.tsv --output out.json
```



## Introduction
### ExplainaBoard is an interpretable, interactive and reliable leaderboard with seven (so far) new features (F) compared with generic leaderboard.
* F1: *Single-system Analysis*: What is a system good or bad at?
* F2: *Pairwise Analysis*: Where is one system better (worse) than another?
* F3: *Data Bias Analysis*: What are the characteristics of different evaluated datasets?
* F5: *Common errors*: What are common mistakes that top-5 systems made?
* F6: *Fine-grained errors*: where will errors occur?
* F7: *System Combination*: Is there potential complementarity between different systems?


<img src="./fig/intro.png" width="400" class="center">






## Website
We deploy ExplainaBoard as a [Web toolkit](http://explainaboard.nlpedia.ai/), which includes 9 NLP tasks, 
40 datasets and 300 systems. Detailed information is as follows.

### Task 

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

<img src="./fig/demo.gif" width="800" class="center">


## Download System Outputs
We haven't released datasets or corresponding system outputs that require licenses. But If you have licenses please fill in this [form](https://docs.google.com/forms/d/1rl7dgOTroT4hazUsd8CaSbGPKFbo2HNOO5pFBsM8IY0/edit) and we will send them to you privately. (Description of output's format can refer [here](https://github.com/neulab/ExplainaBoard/tree/main/output_format)
If these system outputs are useful for you, you can [cite our work](http://explainaboard.nlpedia.ai/explainaboard.bib).




## Submit Your Results
You can submit your system's output by this [form](https://docs.google.com/forms/d/e/1FAIpQLSdb_3PPRTXXjkl9MWUeVLc8Igw0eI-EtOrU93i6B61X9FRJKg/viewform) following the format [description](https://github.com/neulab/ExplainaBoard/tree/main/output_format).



## Acknowledgement
We thanks all authors who share their system outputs with us: Ikuya Yamada, Stefan Schweter,
Colin Raffel, Yang Liu, Li Dong. We also thank
Vijay Viswanathan, Yiran Chen, Hiroaki Hayashi for useful discussion and feedback about ExplainaBoard.

