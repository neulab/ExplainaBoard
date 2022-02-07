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
  <a href=".github/workflows/ci.yml"><img alt="Integration Tests", src="https://github.com/neulab/ExplainaBoard/actions/workflows/ci.yml/badge.svg?event=push" /></a>
</p>

## What is ExplainaBoard?

When developing a natural language processing (NLP or AI) system, often one of the hardest things is to understand where your system is working and where it is failing, and deciding what to do next. ExplainaBoard is a tool that *inspects your system outputs*, *identifies what is working and what is not working*, and helps *inspire you with ideas of where to go next*.

It offers a number of different ways with which you can evaluate and understand your data:

1. *Single-system Analysis*: What is a system good or bad at?
2. *Pairwise Analysis*: Where is one system better (worse) than another?
3. *Data Bias Analysis*: What are the characteristics of different evaluated datasets?
4. *Common Errors*: What are common mistakes that top-5 systems made?
5. *Fine-grained Error Analysis*: where do errors occur?
6. *System Combination*: Is there potential complementarity between different systems?

<img src="./fig/intro.png" width="400" class="center">

How do you get started?

1. *Browse Outputs Online*: We provide an [online interface](http://explainaboard.nlpedia.ai/) where you can browse and understand outputs from various state-of-the-art systems.
2. *Upload Your Own Outputs*: You can also [upload your own system outputs](https://docs.google.com/forms/d/e/1FAIpQLSdb_3PPRTXXjkl9MWUeVLc8Igw0eI-EtOrU93i6B61X9FRJKg/viewform) through the online interace, and we will process them so you can view them alongside other systems.
3. *Run Your Own Analyses Offline*: You can follow the installation instructions below to run analyses on your own system data offline. We also plan to [support command-line upload to the web interface](https://github.com/neulab/ExplainaBoard/issues/60) soon!


## Web-based Toolkit

We have deployed ExplainaBoard as a [Web toolkit](http://explainaboard.nlpedia.ai/), which includes 9 NLP tasks, 
40 datasets and 300 systems. Detailed information is as follows.
<img src="./fig/demo.gif" width="800" class="center">

### Quick Installation

**Method 1:** Simple installation from PyPI (Python 3 only)

```
pip install explainaboard
python -m spacy download en_core_web_sm  # if you plan to use the TextClassificationProcessor
```

**Method 2:** Install from the source and develop locally (Python 3 only)

```bash
# Clone current repo
git clone https://github.com/neulab/ExplainaBoard.git
cd ExplainaBoard

# Install the package
pip install .
```

Then, you can run following examples via bash


### Example for CLI

* text-classification:
```shell
explainaboard --task text-classification --system_outputs ./data/system_outputs/sst2/sst2-lstm.tsv
```
See detailed [description](https://github.com/neulab/ExplainaBoard/blob/main/docs/task_text_classification.md) and 
more [supported tasks](https://github.com/neulab/ExplainaBoard/blob/main/docs/supported_tasks.md).


### Example for Python SDK

```python
from explainaboard import TaskType, get_loader, get_processor

path_data = "./explainaboard/tests/artifacts/test-summ.tsv"
loader = get_loader(TaskType.summarization, data = path_data)
data = loader.load()
processor = get_processor(TaskType.summarization, data = data)
analysis = processor.process()
analysis.write_to_directory("./")
```


## Acknowledgement
We thanks all authors who share their system outputs with us: Ikuya Yamada, Stefan Schweter,
Colin Raffel, Yang Liu, Li Dong. We also thank
Vijay Viswanathan, Yiran Chen, Hiroaki Hayashi for useful discussion and feedback about ExplainaBoard.

