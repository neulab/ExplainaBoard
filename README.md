# ExplainaBoard: An Explainable Leaderboard for NLP
[**Introduction**](#introduction) | 
[**Web Tool**](https://github.com/neulab/ExplainaBoard#web-based-toolkit-quick-learning) |
[**API Tool**](#api-based-toolkit-quick-installation) |
[**Download**](#download-system-outputs) |
[**Paper**](https://arxiv.org/pdf/2104.06387.pdf) |
[**Video**](https://www.youtube.com/watch?v=3X6NgpbN_GU) |
[**Bib**](http://explainaboard.nlpedia.ai/explainaboard.bib)

<p align="center">
  <img src="./fig/logo-full-v2.png" width="800" class="center">
  <br />
  <br />
  <a href="https://github.com/neulab/ExplainaBoard/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/neulab/ExplainaBoard" /></a>
  <a href="https://github.com/neulab/ExplainaBoard/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/neulab/ExplainaBoard" /></a>
  <a href="https://pypi.org/project//"><img alt="PyPI" src="https://img.shields.io/pypi/v/explainaboard" /></a>
  <a href="https://github.com/psf/black"><img alt="Code Style" src="https://img.shields.io/badge/code%20style-black-black" /></a>
</p>











## Introduction
### ExplainaBoard is an interpretable, interactive and reliable leaderboard with seven (so far) new features (F) compared with generic leaderboard.
* F1: *Single-system Analysis*: What is a system good or bad at?
* F2: *Pairwise Analysis*: Where is one system better (worse) than another?
* F3: *Data Bias Analysis*: What are the characteristics of different evaluated datasets?
* F5: *Common errors*: What are common mistakes that top-5 systems made?
* F6: *Fine-grained errors*: where will errors occur?
* F7: *System Combination*: Is there potential complementarity between different systems?


<img src="./fig/intro.png" width="400" class="center">


### Usage
We not only provide a Web-based Interactive Toolkit but also release an API that users can flexible evaluate their systems offline, which
means, you can play with ExplainaBoard at following levels:

* U1: *Just playing with it*: You can walk around, track NLP progress, understand relative merits of different top-performing systems.
* U2: *We help you analyze your model*: You submit your model outputs and deploy them into online ExplainaBoard
* U3: *Do it by yourself*: You can process your model outputs by yourself using our API.



## API-based Toolkit: Quick Installation

#### Method 1: Simple installation from PyPI (Python 3 only)
```
pip install 
```

#### Method 2: Install from the source and develop locally (Python 3 only)
```bash
# Clone current repo
git clone https://github.com/neulab/ExplainaBoard.git
cd ExplainaBoard

# Requirements
pip install -r requirements.txt

# Install the package
python setup.py install
```
#### Then, you can run following examples via bash

```bash
   explainaboard --task chunk --systems ./explainaboard/example/test-conll00.tsv --output out.json
```
where ``test-conll00.tsv`` denotes your system output file whose format depends on different tasks.
For each task we have provided one [example output file](https://github.com/neulab/ExplainaBoard/blob/main/explainaboard/example/) to show how they are formated.
The above command will generate a detailed report (saved in ``out.json``) for your input system (``test-conll00.tsv``).
Specifically, following statistics are included:

* **fine-grained performance**
* **Confidence interval**
* **Error Case**



## Web-based Toolkit: Quick Learning
We deploy ExplainaBoard as a [Web toolkit](http://explainaboard.nlpedia.ai/), which includes 9 NLP tasks, 
40 datasets and 300 systems. Detailed information is as follows.
<img src="./fig/demo.gif" width="800" class="center">


#### So far, ExplainaBoard covers following  tasks 

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






## Submit Your Results
You can submit your system's output by this 
[form](https://docs.google.com/forms/d/e/1FAIpQLSdb_3PPRTXXjkl9MWUeVLc8Igw0eI-EtOrU93i6B61X9FRJKg/viewform) 
following the format [description](https://github.com/neulab/ExplainaBoard/tree/main/output_format).



## Download System Outputs
We haven't released datasets or corresponding system outputs that require licenses. But If you have licenses please fill in this [form](https://docs.google.com/forms/d/1rl7dgOTroT4hazUsd8CaSbGPKFbo2HNOO5pFBsM8IY0/edit) and we will send them to you privately. (Description of output's format can refer [here](https://github.com/neulab/ExplainaBoard/tree/main/output_format)
If these system outputs are useful for you, you can [cite our work](http://explainaboard.nlpedia.ai/explainaboard.bib).


### Currently Covered Systems
So far, ExplainaBoard support more than 10 NLP tasks, including sequence classification, labeling, extraction and generation.
Click [here](http://explainaboard.nlpedia.ai/) to see more.





## Acknowledgement
We thanks all authors who share their system outputs with us: Ikuya Yamada, Stefan Schweter,
Colin Raffel, Yang Liu, Li Dong. We also thank
Vijay Viswanathan, Yiran Chen, Hiroaki Hayashi for useful discussion and feedback about ExplainaBoard.

